# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import mtenv
from gym.vector.async_vector_env import AsyncVectorEnv

from mtrl.env.vec_env import MetaWorldVecEnv, VecEnv
from mtrl.utils.types import ConfigType


import random
from typing import Any, Callable, Dict, List, Optional, Tuple
import metaworld
from gym import Env
from mtenv import MTEnv
from mtenv.envs.metaworld.wrappers.normalized_env import (  # type: ignore[attr-defined]
    NormalizedEnvWrapper,
)
from mtenv.envs.shared.wrappers.multienv import MultiEnvWrapper


def build_dmcontrol_vec_env(
    domain_name: str,
    task_name: str,
    prefix: str,
    make_kwargs: ConfigType,
    env_id_list: List[int],
    seed_list: List[int],
    mode_list: List[str],
) -> VecEnv:
    def get_func_to_make_envs(seed: int, initial_task_state: int):
        def _func() -> mtenv.MTEnv:
            kwargs = deepcopy(make_kwargs)
            kwargs["seed"] += seed
            kwargs["initial_task_state"] = initial_task_state
            return mtenv.make(
                f"MT-HiPBMDP-{domain_name.capitalize()}-{task_name.capitalize()}-vary-{prefix.replace('_', '-')}-v0",
                **kwargs,
            )

        return _func

    funcs_to_make_envs = [
        get_func_to_make_envs(seed=seed, initial_task_state=task_state)
        for (seed, task_state) in zip(seed_list, env_id_list)
    ]

    env_metadata = {"ids": env_id_list, "mode": mode_list}

    env = VecEnv(env_metadata=env_metadata, env_fns=funcs_to_make_envs, context="spawn")

    return env


# copied from here to change the alpha for reward normalisation
# https://github.com/facebookresearch/mtenv/blob/7fdec15f7e842bce4c17f4f3328d9d6fdc79d7fc/mtenv/envs/metaworld/env.py
EnvBuilderType = Callable[[], Env]
TaskStateType = int
TaskObsType = int
EnvIdToTaskMapType = Dict[str, metaworld.Task]
def get_list_of_func_to_make_metaworld_envs(
    benchmark: Optional[metaworld.Benchmark],
    benchmark_name: str,
    env_id_to_task_map: Optional[EnvIdToTaskMapType],
    should_perform_reward_normalization: bool = True,
    task_name: str = "pick-place-v1",
    num_copies_per_env: int = 1,
    reward_alpha=0.001,
) -> Tuple[List[Any], Dict[str, Any]]:
    """Return a list of functions to construct the MetaWorld environments
    and a mapping of environment ids to tasks.
    Args:
        benchmark (Optional[metaworld.Benchmark]): `benchmark` to create
            tasks from.
        benchmark_name (str): name of the `benchmark`. This is used only
            when the `benchmark` is None.
        env_id_to_task_map (Optional[EnvIdToTaskMapType]): In MetaWorld,
            each environment can be associated with multiple tasks. This
            dict persists the mapping between environment ids and tasks.
        should_perform_reward_normalization (bool, optional): Defaults to
            True.
        task_name (str, optional): In case of MT1, only . Defaults to
            "pick-place-v1".
        num_copies_per_env (int, optional): Number of copies to create for
            each environment. Defaults to 1.
    Raises:
        ValueError: if `benchmark` is None and `benchmark_name` is not
            MT1, MT10, or MT50.
    Returns:
        Tuple[List[Any], Dict[str, Any]]: A tuple of two elements. The
        first element is a list of functions to construct the MetaWorld
        environments and the second is a mapping of environment ids
        to tasks.
    """
    if not benchmark:
        if benchmark_name == "MT1":
            benchmark = metaworld.ML1(task_name)
        elif benchmark_name == "MT10":
            benchmark = metaworld.MT10()
        elif benchmark_name == "MT50":
            benchmark = metaworld.MT50()
        else:
            raise ValueError(f"benchmark_name={benchmark_name} is not valid.")

    env_id_list = list(benchmark.train_classes.keys())

    def _get_class_items(current_benchmark):
        return current_benchmark.train_classes.items()

    def _get_tasks(current_benchmark):
        return current_benchmark.train_tasks

    def _get_env_id_to_task_map() -> EnvIdToTaskMapType:
        env_id_to_task_map: EnvIdToTaskMapType = {}
        current_benchmark = benchmark
        for env_id in env_id_list:
            for name, _ in _get_class_items(current_benchmark):
                if name == env_id:
                    task = random.choice(
                        [
                            task
                            for task in _get_tasks(current_benchmark)
                            if task.env_name == name
                        ]
                    )
                    env_id_to_task_map[env_id] = task
        return env_id_to_task_map

    if env_id_to_task_map is None:
        env_id_to_task_map: EnvIdToTaskMapType = _get_env_id_to_task_map()  # type: ignore[no-redef]
    assert env_id_to_task_map is not None

    def get_func_to_make_envs(env_id: str):
        current_benchmark = benchmark

        def _make_env():
            for name, env_cls in _get_class_items(current_benchmark):
                if name == env_id:
                    env = env_cls()
                    task = env_id_to_task_map[env_id]
                    env.set_task(task)
                    if should_perform_reward_normalization:
                        env = NormalizedEnvWrapper(env, normalize_reward=True, reward_alpha=reward_alpha)
                    return env

        return _make_env

    if num_copies_per_env > 1:
        env_id_list = [
            [env_id for _ in range(num_copies_per_env)] for env_id in env_id_list
        ]
        env_id_list = [
            env_id for env_id_sublist in env_id_list for env_id in env_id_sublist
        ]

    funcs_to_make_envs = [get_func_to_make_envs(env_id) for env_id in env_id_list]

    return funcs_to_make_envs, env_id_to_task_map

def build_metaworld_vec_env(
    config: ConfigType,
    benchmark: "metaworld.Benchmark",  # type: ignore[name-defined] # noqa: F821
    mode: str,
    env_id_to_task_map: Optional[Dict[str, "metaworld.Task"]],  # type: ignore[name-defined] # noqa: F821
) -> Tuple[AsyncVectorEnv, Optional[Dict[str, Any]]]:
    benchmark_name = config.env.benchmark._target_.replace("metaworld.", "")
    num_tasks = int(benchmark_name.replace("MT", ""))
    multitask_conf = config.agent.multitask


    make_kwargs = {
        "benchmark": benchmark,
        "benchmark_name": benchmark_name,
        "env_id_to_task_map": env_id_to_task_map,
        "num_copies_per_env": 1,
        "should_perform_reward_normalization": multitask_conf.get('environment_reward_normalisation', False),
        "reward_alpha": config.setup.get("reward_alpha", 0.001),
    }
    funcs_to_make_envs, env_id_to_task_map = get_list_of_func_to_make_metaworld_envs(
        **make_kwargs
    )
    env_metadata = {
        "ids": list(range(num_tasks)),
        "mode": [mode for _ in range(num_tasks)],
    }
    env = MetaWorldVecEnv(
        env_metadata=env_metadata,
        env_fns=funcs_to_make_envs,
        context="spawn",
        shared_memory=False,
    )
    return env, env_id_to_task_map
