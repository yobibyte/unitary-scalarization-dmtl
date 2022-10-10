import torch
import copy
import random
from optimizers.utils import MTLOptimizer


class PCGrad(MTLOptimizer):
    # PCGRad: https://arxiv.org/abs/2001.06782
    def _get_update_direction(self, grads, shared, objectives, shapes=None):
        shared_grads = copy.deepcopy([g_i[shared] for g_i in grads])
        for i, g_i in enumerate(shared_grads):
            indices = list(range(len(grads)))
            random.shuffle(indices)  # Randomly permute indices
            for j in indices:
                if i != j:
                    g_j = grads[j][shared]
                    g_i_g_j = torch.dot(g_i, g_j)
                    if g_i_g_j < 0:
                        g_i -= (g_i_g_j) * g_j / (g_j.norm()**2)
        merged_grad = torch.zeros_like(grads[0]).to(grads[0].device)
        for idx in range(len(grads)):
            merged_grad[shared] += shared_grads[idx]
            merged_grad[~shared] += grads[idx][~shared]
        return merged_grad
