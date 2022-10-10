from typing import List

import torch

from mtrl.agent import grad_manipulation as grad_manipulation_agent
from mtrl.agent import pcgrad as pcgrad_agent
from mtrl.agent.pcgrad import apply_vector_grad_to_parameters
from mtrl.utils.types import TensorType
from optimizers.utils import MTLOptimizer
from optimizers.pcgrad import PCGrad
from optimizers.mgda import MGDA
from optimizers.imtl import IMTL
from optimizers.graddrop import GradDrop
from optimizers.rlw import RLW


def compute_gradient_from_supervised_mtl(
        agent,
        loss: TensorType,  # batch x 1
        parameters: List[TensorType],
        step: int,
        component_names: List[str],
        env_metadata: grad_manipulation_agent.EnvMetadata,
        mtl_optimizer: MTLOptimizer,
        retain_graph: bool = False,
        allow_unused: bool = False
) -> None:
    task_loss = agent._convert_loss_into_task_loss(
        loss=loss, env_metadata=env_metadata
    )
    mtl_optimizer.custom_backwards(task_loss)


class MGDAAgent(pcgrad_agent.Agent):
    """This is the baseline agent that normalizes gradients. """

    def _compute_gradient(
        self,
        loss: TensorType,  # batch x 1
        parameters: List[TensorType],
        step: int,
        component_names: List[str],
        env_metadata: grad_manipulation_agent.EnvMetadata,
        retain_graph: bool = False,
        allow_unused: bool = False,
    ) -> None:

        # a dummy optimizer is created, only used to access parameters
        # mtl_optimizer = MGDA(torch.optim.Adam(parameters), normalize="sig-loss+")  # the loss might be negative in RL

        opt_key = '_'.join(component_names)
        # a dummy optimizer is created, only used to access parameters
        if not hasattr(self, 'mtl_optimizers'):
            self._mtl_optimizers = {}

        if opt_key not in self._mtl_optimizers:
            self._mtl_optimizers[opt_key] = MGDA(torch.optim.Adam(parameters), normalize="l2")  # the loss might be negative in RL

        compute_gradient_from_supervised_mtl(self, loss, parameters, step, component_names, env_metadata,
                                             self._mtl_optimizers[opt_key], retain_graph=retain_graph, allow_unused=allow_unused)


class IMTLAgent(pcgrad_agent.Agent):
    """This is the baseline agent that normalizes gradients. """

    def _compute_gradient(
        self,
        loss: TensorType,  # batch x 1
        parameters: List[TensorType],
        step: int,
        component_names: List[str],
        env_metadata: grad_manipulation_agent.EnvMetadata,
        retain_graph: bool = False,
        allow_unused: bool = False,
    ) -> None:

        # Store SMTO as an attribute to keep state.
        opt_key = '_'.join(component_names)
        if not hasattr(self, 'mtl_optimizers'):
            self._mtl_optimizers = {}

        if opt_key not in self._mtl_optimizers:
            # a dummy optimizer is created, only used to access parameters
            self._mtl_optimizers[opt_key] = IMTL(torch.optim.Adam(parameters), ub=False, numerical_eps=1e-6, learn_scaling=False)

        compute_gradient_from_supervised_mtl(self, loss, parameters, step, component_names, env_metadata,
                                             self._mtl_optimizers[opt_key], retain_graph=retain_graph, allow_unused=allow_unused)


class PCGradWrapperAgent(pcgrad_agent.Agent):

    def _compute_gradient(
        self,
        loss: TensorType,  # batch x 1
        parameters: List[TensorType],
        step: int,
        component_names: List[str],
        env_metadata: grad_manipulation_agent.EnvMetadata,
        retain_graph: bool = False,
        allow_unused: bool = False,
    ) -> None:

        opt_key = '_'.join(component_names)
        if not hasattr(self, 'mtl_optimizers'):
            self._mtl_optimizers = {}

        if opt_key not in self._mtl_optimizers:
            self._mtl_optimizers[opt_key] = PCGrad(torch.optim.Adam(parameters))

        compute_gradient_from_supervised_mtl(self, loss, parameters, step, component_names, env_metadata,
                                             self._mtl_optimizers[opt_key], retain_graph=retain_graph, allow_unused=allow_unused)


class GradDropAgent(pcgrad_agent.Agent):

    def _compute_gradient(
        self,
        loss: TensorType,  # batch x 1
        parameters: List[TensorType],
        step: int,
        component_names: List[str],
        env_metadata: grad_manipulation_agent.EnvMetadata,
        retain_graph: bool = False,
        allow_unused: bool = False,
    ) -> None:

        opt_key = '_'.join(component_names)
        if not hasattr(self, 'mtl_optimizers'):
            self._mtl_optimizers = {}

        if opt_key not in self._mtl_optimizers:
            self._mtl_optimizers[opt_key] = GradDrop(torch.optim.Adam(parameters), use_params_grads=True)

        compute_gradient_from_supervised_mtl(self, loss, parameters, step, component_names, env_metadata,
                                             self._mtl_optimizers[opt_key], retain_graph=retain_graph, allow_unused=allow_unused)


class RLWDirichletAgent(pcgrad_agent.Agent):

    def _compute_gradient(
            self,
            loss: TensorType,  # batch x 1
            parameters: List[TensorType],
            step: int,
            component_names: List[str],
            env_metadata: grad_manipulation_agent.EnvMetadata,
            retain_graph: bool = False,
            allow_unused: bool = False,
    ) -> None:

        opt_key = '_'.join(component_names)
        if not hasattr(self, 'mtl_optimizers'):
            self._mtl_optimizers = {}


        if opt_key not in self._mtl_optimizers:
            self._mtl_optimizers[opt_key] = RLW(torch.optim.Adam(parameters), distribution='dirichlet')

        compute_gradient_from_supervised_mtl(self, loss, parameters, step, component_names, env_metadata,
                                             self._mtl_optimizers[opt_key], retain_graph=retain_graph,
                                             allow_unused=allow_unused)


class RLWNormalAgent(pcgrad_agent.Agent):

    def _compute_gradient(
            self,
            loss: TensorType,  # batch x 1
            parameters: List[TensorType],
            step: int,
            component_names: List[str],
            env_metadata: grad_manipulation_agent.EnvMetadata,
            retain_graph: bool = False,
            allow_unused: bool = False,
    ) -> None:

        opt_key = '_'.join(component_names)
        if not hasattr(self, 'mtl_optimizers'):
            self._mtl_optimizers = {}


        if opt_key not in self._mtl_optimizers:
            self._mtl_optimizers[opt_key] = RLW(torch.optim.Adam(parameters), distribution='normal')

        compute_gradient_from_supervised_mtl(self, loss, parameters, step, component_names, env_metadata,
                                             self._mtl_optimizers[opt_key], retain_graph=retain_graph,
                                             allow_unused=allow_unused)
