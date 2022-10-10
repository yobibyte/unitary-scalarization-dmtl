import torch
import numpy as np
from optimizers.utils import MTLOptimizer, batch_average, standard_head_backward


class MGDA(MTLOptimizer):
    # MGDA: https://arxiv.org/pdf/1810.04650
    def __init__(self, optimizer, specialized_parameters=None, scheduler=None, normalize="loss+", ub=False):
        # normalize: normalization type, see function normalize_gradients. loss and loss+ require non-negative losses
        # loss+ is the default as in https://github.com/isl-org/MultiObjectiveOptimization/blob/master/sample.json
        super().__init__(optimizer, scheduler=scheduler)
        assert normalize in ["l2", "loss", "loss+", "sig-loss+", "none"]
        self.normalize = normalize
        self.ub = ub
        self.shared_repr = None
        self.specialized_params = specialized_parameters
        self._alpha_to_log = None

    def _get_update_direction(self, grads, shared, objectives, shapes=None, return_alpha=False):
        if not return_alpha:
            merged_grad = torch.zeros_like(grads[0]).to(grads[0].device)
            for idx in range(len(grads)):
                # Use plain gradients for task-specific parameters
                merged_grad[~shared] += grads[idx][~shared].clone()
            shared_grads = [g_i[shared] for g_i in grads]
        else:
            shared_grads = grads

        # Normalize all gradients, this is optional and not included in the paper.
        normalize_gradients(shared_grads, objectives, self.normalize)

        with torch.no_grad():
            # Compute convex combination coefficients on gradients of shared parameters.
            # TODO: in the paper they claim to use FW, but their code uses projected gradient descent.
            #  the authors provide find_min_norm_element_FW but do not seem to use it
            alpha, min_norm = MinNormSolver.find_min_norm_element(shared_grads)
        self._alpha_to_log = alpha
        if return_alpha:
            return alpha

        for idx in range(len(grads)):
            # Compute the affine combination for shared parameters
            merged_grad[shared] += shared_grads[idx].mul_(alpha[idx])
        return merged_grad

    def custom_backwards(self, objectives):
        spec_params_flag = (self.specialized_params is not None)
        # Allow to switch between MGDA and MGDA-UB (see original MGDA paper)
        if not self.ub:
            # MGDA
            MTLOptimizer.custom_backwards(self, objectives)
        else:
            # MGDA-UB
            objectives = batch_average(objectives)
            # Use approximation given by the gradient of the loss w.r.t. shared parameters to find convex combination
            # coefficients, then backward on the scaled sum
            z_grads = []
            for obj in objectives:
                z_grads.append(torch.autograd.grad(obj, self.shared_repr, only_inputs=True, retain_graph=True)[0])
            self.shared_repr = None
            # the implementation of find_min_norm_element implicitly linearizes the gradient of z
            # (which is of size batch_size x repr_size)
            alpha = self._get_update_direction(z_grads, None, objectives, return_alpha=True)
            del z_grads
            mgda_objective = sum([obj * alpha[idx] for idx, obj in enumerate(objectives)])
            mgda_objective.backward(retain_graph=spec_params_flag)

            if spec_params_flag:
                # Overwrite gradients of specialized parameters
                standard_head_backward(objectives, self.specialized_params)

    def set_auxiliaries(self, **kwargs):
        # Pass any auxiliary parameter.
        if self.ub:
            self.shared_repr = kwargs["shared_repr"]


class MinNormSolver:
    # Code adapted from  https://github.com/isl-org/MultiObjectiveOptimization
    MAX_ITER = 250
    STOP_CRIT = 1e-5

    @staticmethod
    def _min_norm_element_from2(v1v1, v1v2, v2v2):
        """
        Analytical solution for min_{c} |cx_1 + (1-c)x_2|_2^2
        d is the distance (objective) optimzed
        v1v1 = <x1,x1>
        v1v2 = <x1,x2>
        v2v2 = <x2,x2>
        """
        # TODO: why 0.999 and 0.001, and not plain 0 and 1? Do the inner problems ever converge, like this?
        if v1v2 >= v1v1:
            # Case: Fig 1, third column
            gamma = 0.999
            cost = v1v1
            return gamma, cost
        if v1v2 >= v2v2:
            # Case: Fig 1, first column
            gamma = 0.001
            cost = v2v2
            return gamma, cost
        # Case: Fig 1, second column
        gamma = -1.0 * ((v1v2 - v2v2) / (v1v1 + v2v2 - 2 * v1v2))
        cost = v2v2 + gamma * (v1v2 - v2v2)
        return gamma, cost

    @staticmethod
    def _min_norm_2d(vecs, dps):
        """
        Find the minimum norm solution as combination of two points
        This is correct only in 2D
        ie. min_c |\sum c_i x_i|_2^2 st. \sum c_i = 1 , 1 >= c_1 >= 0 for all i, c_i + c_j = 1.0 for some i, j
        """
        dmin = float("inf")
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                if (i, j) not in dps:
                    dps[(i, j)] = torch.mul(vecs[i], vecs[j]).sum().data.item()
                    dps[(j, i)] = dps[(i, j)]
                if (i, i) not in dps:
                    dps[(i, i)] = torch.mul(vecs[i], vecs[i]).sum().data.item()
                if (j, j) not in dps:
                    dps[(j, j)] = torch.mul(vecs[j], vecs[j]).sum().data.item()
                c, d = MinNormSolver._min_norm_element_from2(dps[(i, i)], dps[(i, j)], dps[(j, j)])
                if d < dmin:
                    dmin = d
                    sol = [(i, j), c, d]
        return sol, dps

    @staticmethod
    def _projection2simplex(y):
        """
        Given y, it solves argmin_z |y-z|_2 st \sum z = 1 , 1 >= z_i >= 0 for all i
        """
        m = len(y)
        sorted_y = np.flip(np.sort(y), axis=0)
        tmpsum = 0.0
        tmax_f = (np.sum(y) - 1.0) / m
        for i in range(m - 1):
            tmpsum += sorted_y[i]
            tmax = (tmpsum - 1) / (i + 1.0)
            if tmax > sorted_y[i + 1]:
                tmax_f = tmax
                break
        return np.maximum(y - tmax_f, np.zeros(y.shape))

    @staticmethod
    def _next_point(cur_val, grad, n):
        # note: this should be part of the routine projecting to the simplex
        proj_grad = grad - (np.sum(grad) / n)
        tm1 = -1.0 * cur_val[proj_grad < 0] / proj_grad[proj_grad < 0]
        tm2 = (1.0 - cur_val[proj_grad > 0]) / (proj_grad[proj_grad > 0])

        t = 1
        if len(tm1[tm1 > 1e-7]) > 0:
            t = np.min(tm1[tm1 > 1e-7])
        if len(tm2[tm2 > 1e-7]) > 0:
            t = min(t, np.min(tm2[tm2 > 1e-7]))

        next_point = proj_grad * t + cur_val
        next_point = MinNormSolver._projection2simplex(next_point)
        return next_point

    @staticmethod
    def find_min_norm_element(vecs):
        """
        Given a list of vectors (vecs), this method finds the minimum norm element in the convex hull
        as min |u|_2 st. u = \sum c_i vecs[i] and \sum c_i = 1.
        It is quite geometric, and the main idea is the fact that if d_{ij} = min |u|_2 st u = c x_i + (1-c) x_j;
        the solution lies in (0, d_{i,j}).
        Hence, we find the best 2-task solution, and then run the projected gradient descent until convergence
        Note: the PGD step is not taken directly: the optimal point on the convex combination between the step and
        the current iterate is taken.
        """
        # Solution lying at the combination of two points
        dps = {}
        init_sol, dps = MinNormSolver._min_norm_2d(vecs, dps)

        n = len(vecs)
        sol_vec = np.zeros(n)
        sol_vec[init_sol[0][0]] = init_sol[1]
        sol_vec[init_sol[0][1]] = 1 - init_sol[1]

        if n < 3:
            # This is optimal for n=2, so return the solution
            return sol_vec, init_sol[2]

        iter_count = 0

        grad_mat = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                grad_mat[i, j] = dps[(i, j)]

        while iter_count < MinNormSolver.MAX_ITER:
            grad_dir = -1.0 * np.dot(grad_mat, sol_vec)
            new_point = MinNormSolver._next_point(sol_vec, grad_dir, n)
            # Re-compute the inner products for line search
            v1v1 = 0.0
            v1v2 = 0.0
            v2v2 = 0.0
            for i in range(n):
                for j in range(n):
                    v1v1 += sol_vec[i] * sol_vec[j] * dps[(i, j)]
                    v1v2 += sol_vec[i] * new_point[j] * dps[(i, j)]
                    v2v2 += new_point[i] * new_point[j] * dps[(i, j)]
            nc, nd = MinNormSolver._min_norm_element_from2(v1v1, v1v2, v2v2)
            new_sol_vec = nc * sol_vec + (1 - nc) * new_point
            change = new_sol_vec - sol_vec
            if np.sum(np.abs(change)) < MinNormSolver.STOP_CRIT:
                return sol_vec, nd
            sol_vec = new_sol_vec
            iter_count += 1
        return sol_vec, nd

    @staticmethod
    def find_min_norm_element_FW(vecs):
        """
        Given a list of vectors (vecs), this method finds the minimum norm element in the convex hull
        as min |u|_2 st. u = \sum c_i vecs[i] and \sum c_i = 1.
        It is quite geometric, and the main idea is the fact that if d_{ij} = min |u|_2 st u = c x_i + (1-c) x_j;
        the solution lies in (0, d_{i,j}).
        Hence, we find the best 2-task solution, and then run the Frank Wolfe until convergence
        """
        # Solution lying at the combination of two points
        dps = {}
        init_sol, dps = MinNormSolver._min_norm_2d(vecs, dps)

        n = len(vecs)
        sol_vec = np.zeros(n)
        sol_vec[init_sol[0][0]] = init_sol[1]
        sol_vec[init_sol[0][1]] = 1 - init_sol[1]

        if n < 3:
            # This is optimal for n=2, so return the solution
            return sol_vec, init_sol[2]

        iter_count = 0

        grad_mat = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                grad_mat[i, j] = dps[(i, j)]

        while iter_count < MinNormSolver.MAX_ITER:
            t_iter = np.argmin(np.dot(grad_mat, sol_vec))

            v1v1 = np.dot(sol_vec, np.dot(grad_mat, sol_vec))
            v1v2 = np.dot(sol_vec, grad_mat[:, t_iter])
            v2v2 = grad_mat[t_iter, t_iter]

            nc, nd = MinNormSolver._min_norm_element_from2(v1v1, v1v2, v2v2)
            new_sol_vec = nc * sol_vec
            new_sol_vec[t_iter] += 1 - nc

            change = new_sol_vec - sol_vec
            if np.sum(np.abs(change)) < MinNormSolver.STOP_CRIT:
                return sol_vec, nd
            sol_vec = new_sol_vec
            iter_count += 1


def normalize_gradients(grads, losses, normalization_type):
    # Code adapted from  https://github.com/isl-org/MultiObjectiveOptimization
    for idx, g_i in enumerate(grads):
        if normalization_type == 'l2':
            denom = g_i.norm()
        elif normalization_type == 'loss':
            denom = losses[idx]
        elif normalization_type == 'loss+':
            denom = losses[idx] * g_i.norm()
        elif normalization_type == 'sig-loss+':
            # Not in original paper, useful for non-
            denom = torch.sigmoid(losses[idx]) * g_i.norm()
        elif normalization_type == 'none':
            denom = 1.0
        else:
            print('ERROR: Invalid Normalization Type')
        if denom != 0:
            g_i /= denom