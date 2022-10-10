#!/bin/bash
# Use the first argument to provide the seed.
PYTHONPATH=$PYTHONPATH:../..:. python3.8 -u main.py \
setup=metaworld \
env=metaworld-mt10 \
agent=pcgrad_state_sac \
experiment.num_eval_episodes=1 \
experiment.num_train_steps=2000000 \
setup.seed=1 \
replay_buffer.batch_size=1280 \
agent.multitask.num_envs=10 \
agent.multitask.should_use_disentangled_alpha=False \
agent.multitask.should_use_task_encoder=False \
agent.multitask.actor_cfg.should_condition_encoder_on_task_info=False \
agent.multitask.actor_cfg.should_concatenate_task_info_with_encoder=False \
agent.encoder.type_to_select=identity \
+label=pcgrad_old_v2 \
experiment.save.model.retain_last_n=-1 \
setup.seed=$1
