#!/bin/bash
# Use the first argument to provide the seed.
PYTHONPATH=$PYTHONPATH:../..:. python3.8 -u main.py \
setup=metaworld \
env=metaworld-mt10 \
agent=state_sac \
experiment.num_eval_episodes=1 \
experiment.num_train_steps=2000000 \
replay_buffer.batch_size=1280 \
replay_buffer.capacity=4000000 \
agent.multitask.num_envs=10 \
agent.multitask.should_use_disentangled_alpha=False \
agent.encoder.type_to_select=identity \
agent.multitask.should_use_multi_head_policy=False \
experiment.save.model.retain_last_n=1 \
+agent.multitask.buffer_reward_normalisation=True \
+agent.multitask.environment_reward_normalisation=False \
+agent.multitask.clip_targets=False \
setup.seed=$1 \
+label=mtsac_buffer_rewnorm_4x_buf_share_alpha
