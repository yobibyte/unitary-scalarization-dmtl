#!/bin/bash
# Use the first argument to provide the seed.
PYTHONPATH=$PYTHONPATH:../..:. python3.8 -u main.py \
setup=metaworld \
env=metaworld-mt10 \
agent=imtl_from_mtl_state_sac \
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
+agent.multitask.clip_targets=True \
+agent.multitask.alpha_use_naive_opt=True \
+agent.optimizers.actor.weight_decay=3e-4 \
setup.seed=$1 \
+label=imtl_mt10_buffer_rewnorm_target_clip_4x_buf_naive_alpha_no_scaling_actor_reg_3em4
