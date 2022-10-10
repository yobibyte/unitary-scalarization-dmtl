# Code adapted from https://github.com/tensorflow/lingvo/blob/master/lingvo/core/graddrop.py

import torch
from optimizers.utils import MTLOptimizer, standard_head_backward, batch_average


class GradDrop(MTLOptimizer):
    # GradDrop: https://arxiv.org/abs/2010.06808
    # Note: all leaks are set to 0: "Pure GradDrop, see Algorithm 1 in paper"
    def __init__(self, optimizer, specialized_parameters=None, scheduler=None, k=1.0, use_sign=True, p=0.5,
                 use_params_grads=False):
        # If use_params_grads is true, doesn't implement graddrop as a layer, but over the gradients w.r.t. params
        super().__init__(optimizer, scheduler=scheduler)
        self.k = k
        self.p = p  # used only if use_sign is False
        self.use_sign = use_sign
        self.shared_repr = None
        self.use_parameter_grads = use_params_grads
        self.specialized_params = specialized_parameters
        self.epsilon = 1e-7

    def custom_backwards(self, objectives):
        if self.use_parameter_grads:
            MTLOptimizer.custom_backwards(self, objectives)
        else:
            spec_params_flag = (self.specialized_params is not None)
            objectives = batch_average(objectives)
            # Compute gradient of activations at shared representation.
            a_grads = []
            for obj in objectives:
                a_grads.append(torch.autograd.grad(obj, self.shared_repr, only_inputs=True, retain_graph=True)[0])

            gd_out = self._graddrop_update_direction(a_grads)
            # gd_out acts as the new jacobian-vector product for the backward pass for the parameter's gradients.
            torch.autograd.backward(self.shared_repr, grad_tensors=gd_out, retain_graph=spec_params_flag)
            self.shared_repr = None

            if spec_params_flag:
                # Compute gradients of specialized parameters (not computed by backward above)
                standard_head_backward(objectives, self.specialized_params)

    def set_auxiliaries(self, **kwargs):
        # Pass any auxiliary parameter.
        self.shared_repr = kwargs["shared_repr"]

    def _get_update_direction(self, grads, shared, objectives, shapes=None, return_alpha=False):
        merged_grad = torch.zeros_like(grads[0]).to(grads[0].device)
        for idx in range(len(grads)):
            # Use plain gradients for task-specific parameters
            merged_grad[~shared] += grads[idx][~shared].clone()
        shared_grads = [g_i[shared] for g_i in grads]
        # Normalize all gradients, this is optional and not included in the paper.
        merged_grad[shared] = self._graddrop_update_direction(shared_grads)
        return merged_grad

    def _graddrop_update_direction(self, grads):
        sum_a_grad = sum(grads)

        if not self.use_parameter_grads:
            # Multiply the gradients with the input's sign (epsilon is used by the authors to ignore small activations)
            input_abs = torch.abs((torch.abs(self.shared_repr) <= self.epsilon).float() + self.shared_repr)
            G = [grad * ((self.shared_repr) / (input_abs)) for grad in grads]
            # Sum gradient over batch.
            G = [grad.sum(dim=0) for grad in G]
        else:
            # Section 3.2 of the GradDrop paper does not apply if the algorithm is implemented over parameter grads.
            G = grads

        if self.use_sign:
            # GradDrop, samples so that the signs are coherent.
            # First discretize all gradients into their sign values.
            grad_sign_positive = [(grad > 0.0).float() for grad in G]
            grad_sign_negative = [(grad < 0.0).float() for grad in G]

            # Calculate the probability of positive gradients based on equation (1)
            # in the GradDrop paper.
            grad_abs_sum = sum([torch.abs(grad) for grad in G])
            # Implementation of different scales for the keep function. Larger scales result in steeper keep functions.
            prob_pos = self.k * (sum(G) / (2. * grad_abs_sum + self.epsilon)) + 0.5

            # The main, default mode of GradDrop. Only gradients of one sign are kept,
            # and which sign is calculated via equation (1) of the main paper.
            prob_pos = (prob_pos >= torch.rand(prob_pos.shape, device=grads[0].device)).float() - 0.5
            grad_masks = [(gsp - gsn) * prob_pos >= 0 for (gsn, gsp) in zip(grad_sign_negative, grad_sign_positive)]
        else:
            # Random GradDrop, samples entries masks with fixed probability regardless of the gradient signs.
            # NOTE: this differs from the GradDrop paper's "Random GradDrop" implementation, where only the probability is
            # independent of the sign, but only a single sign remains unmasked
            grad_masks = [torch.bernoulli((1 - self.p) * torch.ones_like(c_grad)) for c_grad in grads]

        transformed_grad = sum([grad * grad_mask.float() for (grad, grad_mask) in zip(grads, grad_masks)])

        # Re-normalize so that the final graddrop norm is the same as the original gradient's
        transformed_grad_norm = transformed_grad.norm()
        original_grad_norm = sum_a_grad.norm()
        gd_out = transformed_grad * original_grad_norm / (transformed_grad_norm + self.epsilon)
        return gd_out
