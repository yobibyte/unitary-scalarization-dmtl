import torch
from optimizers.utils import MTLOptimizer, batch_average, standard_head_backward


class IMTL(MTLOptimizer):
    # IMTL: https://openreview.net/forum?id=IMPnRXEWpvr
    def __init__(self, optimizer, specialized_parameters=None, scheduler=None, ub=True, learn_scaling=True, numerical_eps=1e-6):
        # UB=True means that the IMTL-G part of the algorithm is applied on the gradient of the shared representation
        # (analogously to MGDA-UB)
        # learn_scaling=False disables IMTL-L
        super().__init__(optimizer, scheduler=scheduler)
        self.ub = ub
        self.shared_repr = None
        self.st = None
        self.learn_scaling = learn_scaling
        self.st_optimizer = None
        self.specialized_params = specialized_parameters
        self.numerical_eps = numerical_eps
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

        # Build matrices of gradient differences.
        D = (shared_grads[0] - shared_grads[1]).unsqueeze(-1)
        u_0 = shared_grads[0]/shared_grads[0].norm()
        U = (u_0 - shared_grads[1]/shared_grads[1].norm()).unsqueeze(-1)
        for g_i in shared_grads[2:]:
            D = torch.cat([D, (shared_grads[0] - g_i).unsqueeze(-1)], -1)
            U = torch.cat([U, (u_0 - g_i/g_i.norm()).unsqueeze(-1)], -1)

        # Compute minimal-norm affine combination coefficients.
        A = D.transpose(-2, -1) @ U
        A += self.numerical_eps * torch.eye(A.shape[-1], device=A.device)   # avoid singularity
        alpha = torch.inverse(A) @ torch.mv(U.transpose(-2, -1), shared_grads[0])
        alpha = torch.cat([1 - alpha.sum().unsqueeze(-1), alpha], 0)
        self._alpha_to_log = alpha
        if torch.isnan(alpha).any():
            # Some gradient was 0: the solution of the minimal-norm affine combination is 0.
            alpha = torch.zeros_like(alpha)
        if return_alpha:
            return alpha

        for idx in range(len(grads)):
            # Compute the affine combination for shared parameters
            merged_grad[shared] += shared_grads[idx].mul_(alpha[idx])
        return merged_grad

    def custom_backwards(self, objectives):
        objectives = batch_average(objectives)

        if self.learn_scaling:
            # IMTL-L part.
            # using the initial step size of the optimizer as step size for vanilla SGD: detail not specified in paper
            if self.st == None:
                self.st = torch.zeros((len(objectives),) + tuple(objectives[0].shape), requires_grad=True,
                                      device=objectives[0].device)
                self.st_optimizer = torch.optim.Adam([self.st], lr=self._optim.defaults['lr'])
            scaled_losses = [torch.exp(self.st[idx]) * objectives[idx] - self.st[idx] for idx in range(len(objectives))]
        else:
            scaled_losses = objectives

        if not self.ub:
            # Apply IMTL-G on the gradient of the shared parameters (tasks-specific parameters, including self.st,
            # are updated using the scaled objective sum)
            MTLOptimizer.custom_backwards(self, scaled_losses)
        else:
            # Apply IMTL-G on the gradient of the shared representation
            # (tasks-specific parameters, including self.st, are updated using the scaled objective sum)
            # Use approximation given by the gradient of the loss w.r.t. shared parameters to find convex combination
            # coefficients, then backward on the scaled sum
            z_grads = []
            for obj in scaled_losses:
                # the z_grads are different for each batch entry: this dimension is linearized as suggested
                # by the MGDA authors (detail not specified in the IMTL paper)
                z_grads.append(
                    torch.autograd.grad(obj, self.shared_repr, only_inputs=True, retain_graph=True)[0].view(-1))
            self.shared_repr = None
            alpha = self._get_update_direction(z_grads, None, scaled_losses, return_alpha=True)
            del z_grads
            objective = sum([obj * alpha[idx] for idx, obj in enumerate(scaled_losses)])
            objective.backward(retain_graph=True)

            # standard_head_backward is called on the specialized parameters as well as on self.st (if learning it)
            standard_sgd_params = self.specialized_params + [self.st] if self.learn_scaling else self.specialized_params
            standard_head_backward(scaled_losses, specialized_params=standard_sgd_params)

        if self.learn_scaling:
            self.st_optimizer.step()  # the grad is accumulated above
            self.st_optimizer.zero_grad()

    def set_auxiliaries(self, **kwargs):
        # Pass any auxiliary parameter.
        if self.ub:
            self.shared_repr = kwargs["shared_repr"]
