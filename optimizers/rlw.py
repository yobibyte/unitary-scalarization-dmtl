import torch
import random
import torch.nn.functional as F
from optimizers.utils import MTLOptimizer


class RLW(MTLOptimizer):
    # https://openreview.net/forum?id=OdnNBNIdFul (adapted from their ICLR22 code release)

    def __init__(self, optimizer, m=None, scheduler=None, distribution="uniform"):
        super().__init__(optimizer, scheduler=scheduler)
        self.distribution = distribution
        self.random_normal_mean = None if m is None else torch.rand(m)
        self.random_normal_std = None if m is None else torch.rand(m)

    def custom_backwards(self, objectives):

        # If the loss hasn't been averaged for the mini-batch, do it now.
        if objectives[0].dim() > 0 and objectives[0].shape[0] > 1:
            objectives = [obj.mean(dim=0) for obj in objectives]

        # Sample weights.
        m = len(objectives)
        if self.distribution == 'uniform':
            batch_weight = F.softmax(torch.rand(m).to(objectives[0].device), dim=-1)
        elif self.distribution == 'normal':
            batch_weight = F.softmax(torch.randn(m).to(objectives[0].device), dim=-1)
        elif self.distribution == 'dirichlet':
            # https://en.wikipedia.org/wiki/Dirichlet_distribution#Random_number_generation
            alpha = 1
            gamma_sample = [random.gammavariate(alpha, 1) for _ in range(m)]
            dirichlet_sample = [v / sum(gamma_sample) for v in gamma_sample]
            batch_weight = torch.tensor(dirichlet_sample).to(objectives[0].device)
        elif self.distribution == 'random_normal':
            if self.random_normal_mean is None:
                self.random_normal_mean = torch.rand(len(objectives))
                self.random_normal_std  = torch.rand(len(objectives))
            batch_weight = F.softmax(
                torch.normal(self.random_normal_mean, self.random_normal_std).to(objectives[0].device), dim=-1)
        elif self.distribution == 'bernoulli':
            while True:
                w = torch.randint(0, 2, (m,))
                if w.sum() != 0:
                    batch_weight = w.to(objectives[0].device).float()
                    break
        elif self.distribution == 'constrained_bernoulli':
            w = random.sample(range(m), k=1)
            batch_weight = torch.zeros(m).to(objectives[0].device)
            batch_weight[w] = 1.
        else:
            raise ('no support {}'.format(self.distribution))

        # Normalize to m * simplex
        batch_weight *= m / batch_weight.sum()

        objective = torch.zeros_like(objectives[0])
        for idx, obj in enumerate(objectives):
            objective += batch_weight[idx] * obj
        objective.backward()
