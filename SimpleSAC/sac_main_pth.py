import datetime
import os
import sys
import time
from copy import deepcopy
import uuid

import numpy as np
import pprint

import gym
import torch

import absl.app
import absl.flags
import wandb
from tqdm import trange

from sac import SAC
from envs import get_new_density_env, get_new_friction_env, get_new_gravity_env, get_new_thigh_range_env, get_new_foot_shape_env, get_new_foot_stiffness_env, get_new_thigh_size_env, get_new_ellipsoid_limb_env, get_new_box_limb_env, get_new_head_size_env, get_new_torso_length_env, get_new_limb_stiffness_env, get_new_tendon_elasticity_env
from replay_buffer import ReplayBuffer, batch_to_torch
from model import TanhGaussianPolicy, FullyConnectedQFunction, SamplerPolicy
from sampler import StepSampler, TrajSampler
from utils import Timer, define_flags_with_default, set_random_seed, print_flags, get_user_flags, prefix_metrics
from utils import WandBLogger

sys.path.append("..")
from viskit.logging import logger, setup_logger

nowTime = datetime.datetime.now().strftime('%y-%m-%d-%H-%M-%S')

FLAGS_DEF = define_flags_with_default(
    current_time=nowTime,
    name_str='',
    env_list='Hopper-v2',
    data_source='medium_replay',
    unreal_dynamics="gravity",
    variety_list="2.0",
    sim_only=False,
    penalize_sim=True,
    sim_warmup=0,
    max_traj_length=1000,
    replay_buffer_size=1000000,
    seed=42,
    device='cuda',
    save_model=True,
    save_replaybuffer=False,

    reward_scale=1.0,
    reward_bias=0.0,
    clip_action=1.0,
    joint_noise_std=0.0,

    policy_arch='256-256',
    qf_arch='256-256',
    orthogonal_init=False,
    policy_log_std_multiplier=1.0,
    policy_log_std_offset=-1.0,

    n_epochs=1000,
    n_env_steps_per_epoch=1000,
    n_train_step_per_epoch=1000,
    eval_period=10,
    eval_n_trajs=5,

    batch_size=256,

    sac=SAC.get_default_config(),
    logging=WandBLogger.get_default_config(),
)


def main(argv):
    FLAGS = absl.flags.FLAGS

    variant = get_user_flags(FLAGS, FLAGS_DEF)
    wandb_logger = WandBLogger(config=FLAGS.logging, variant=variant)
    wandb.run.name = f"{FLAGS.name_str}_{FLAGS.env_list}_{FLAGS.data_source}_{FLAGS.unreal_dynamics}x{FLAGS.variety_list}_seed={FLAGS.seed}_{FLAGS.current_time}"
    setup_logger(
        variant=variant,
        exp_id=wandb_logger.experiment_id,
        seed=FLAGS.seed,
        base_log_dir=FLAGS.logging.output_dir,
        include_exp_prefix_sub_dir=False
    )

    set_random_seed(FLAGS.seed)

    # different unreal dynamics properties: gravity; density; friction
    for unreal_dynamics in FLAGS.unreal_dynamics.split(";"):
        # different environment: Walker2d-v2, Hopper-v2, HalfCheetah-v2
        for env_name in FLAGS.env_list.split(";"):
            # different varieties: 0.5, 1.5, 2.get_new_foot_stiffness_env0, ...
            for variety_degree in FLAGS.variety_list.split(";"):
                variety_degree = float(variety_degree)

                if env_name == "Humanoid-v2":
                    off_env_name = env_name
                else:
                    off_env_name = "{}-{}-{}".format(env_name.split("-")[0].lower(), FLAGS.data_source, env_name.split("-")[-1]).replace('_',"-")
                if unreal_dynamics == "gravity":
                    real_env = get_new_gravity_env(1, off_env_name)
                    sim_env = get_new_gravity_env(variety_degree, off_env_name)
                elif unreal_dynamics == "density":
                    real_env = get_new_density_env(1, off_env_name)
                    sim_env = get_new_density_env(variety_degree, off_env_name)
                elif unreal_dynamics == "friction":
                    real_env = get_new_friction_env(1, off_env_name)
                    sim_env = get_new_friction_env(variety_degree, off_env_name)
                elif unreal_dynamics == "broken_thigh":
                    real_env = get_new_thigh_range_env(1, off_env_name)
                    sim_env = get_new_thigh_range_env(variety_degree, off_env_name)
                elif unreal_dynamics == "ellipsoid_foot":
                    real_env = get_new_gravity_env(1, off_env_name)
                    sim_env =  get_new_foot_shape_env(off_env_name)
                elif unreal_dynamics == "soft_foot":
                    real_env = get_new_foot_stiffness_env(1, off_env_name)
                    sim_env = get_new_foot_stiffness_env(variety_degree, off_env_name)
                elif unreal_dynamics == "soft_limb":
                    real_env = get_new_limb_stiffness_env(1, off_env_name)
                    sim_env = get_new_limb_stiffness_env(variety_degree, off_env_name)
                elif unreal_dynamics == "elastic_tendon":
                    real_env = get_new_tendon_elasticity_env(1, off_env_name)
                    sim_env = get_new_tendon_elasticity_env(variety_degree, off_env_name)
                elif unreal_dynamics == "thigh_size":
                    real_env = get_new_thigh_size_env(1, off_env_name)
                    sim_env = get_new_thigh_size_env(variety_degree, off_env_name)
                elif unreal_dynamics == "ellipsoid_limb":
                    real_env = get_new_gravity_env(1, off_env_name)
                    sim_env =  get_new_ellipsoid_limb_env(off_env_name)
                elif unreal_dynamics == "box_limb":
                    real_env = get_new_gravity_env(1, off_env_name)
                    sim_env =  get_new_box_limb_env(off_env_name)
                elif unreal_dynamics == "head_size":
                    real_env = get_new_head_size_env(1, off_env_name)
                    sim_env = get_new_head_size_env(variety_degree, off_env_name)
                elif unreal_dynamics == "torso_length":
                    real_env = get_new_torso_length_env(1, off_env_name)
                    sim_env = get_new_torso_length_env(variety_degree, off_env_name)
                else:
                    raise RuntimeError("Got erroneous unreal dynamics %s" % unreal_dynamics)

                print("\n-------------Env name: {}, variety: {}, unreal_dynamics: {}-------------".format(env_name, variety_degree, unreal_dynamics))

    # a step sampler for "simulated" training
    train_sampler = StepSampler(sim_env.unwrapped, FLAGS.max_traj_length)
    # a trajectory sampler for "real-world" evaluation
    eval_sampler = TrajSampler(real_env.unwrapped, FLAGS.max_traj_length)

    num_state = real_env.observation_space.shape[0]
    num_action = real_env.action_space.shape[0]
    replay_buffer = ReplayBuffer(num_state, num_action, FLAGS.replay_buffer_size)

    policy = TanhGaussianPolicy(
        eval_sampler.env.observation_space.shape[0],
        eval_sampler.env.action_space.shape[0],
        arch=FLAGS.policy_arch,
        log_std_multiplier=FLAGS.policy_log_std_multiplier,
        log_std_offset=FLAGS.policy_log_std_offset,
        orthogonal_init=FLAGS.orthogonal_init,
    )

    qf1 = FullyConnectedQFunction(
        eval_sampler.env.observation_space.shape[0],
        eval_sampler.env.action_space.shape[0],
        arch=FLAGS.qf_arch,
        orthogonal_init=FLAGS.orthogonal_init,
    )
    target_qf1 = deepcopy(qf1)

    qf2 = FullyConnectedQFunction(
        eval_sampler.env.observation_space.shape[0],
        eval_sampler.env.action_space.shape[0],
        arch=FLAGS.qf_arch,
        orthogonal_init=FLAGS.orthogonal_init,
    )
    target_qf2 = deepcopy(qf2)

    if FLAGS.sac.target_entropy >= 0.0:
        FLAGS.sac.target_entropy = -np.prod(eval_sampler.env.action_space.shape).item()

    sac = SAC(FLAGS.sac, policy, qf1, qf2, target_qf1, target_qf2)
    sac.torch_to_device(FLAGS.device)

    sampler_policy = SamplerPolicy(policy, FLAGS.device)

    viskit_metrics = {}
    for epoch in trange(FLAGS.n_epochs):
        metrics = {}
        with Timer() as rollout_timer:
            train_sampler.sample(
                sampler_policy, FLAGS.n_env_steps_per_epoch,
                deterministic=False, replay_buffer=replay_buffer, joint_noise_std=FLAGS.joint_noise_std
            )
            # metrics['env_steps'] = replay_buffer.total_steps
            metrics['epoch'] = epoch

        with Timer() as train_timer:
            for batch_idx in trange(FLAGS.n_train_step_per_epoch):
                batch = replay_buffer.sample(FLAGS.batch_size)
                if batch_idx + 1 == FLAGS.n_train_step_per_epoch:
                    metrics.update(
                        prefix_metrics(sac.train(batch), 'sac')
                    )
                else:
                    sac.train(batch)

        with Timer() as eval_timer:
            if epoch == 0 or (epoch + 1) % FLAGS.eval_period == 0:
                trajs = eval_sampler.sample(
                    sampler_policy, FLAGS.eval_n_trajs, deterministic=True
                )

                metrics['average_return'] = np.mean([np.sum(t['rewards']) for t in trajs])
                metrics['average_traj_length'] = np.mean([len(t['rewards']) for t in trajs])
                # metrics['average_normalizd_return'] = np.mean(
                #     [eval_sampler.env.get_normalized_score(np.sum(t['rewards'])) for t in trajs]
                # )

                if FLAGS.save_model:
                    save_data = {'sac': sac, 'variant': variant, 'epoch': epoch}
                    wandb_logger.save_pickle(save_data, '{}_{}x{}_model.pkl'.format(FLAGS.env_list, FLAGS.unreal_dynamics, FLAGS.variety_list))
                    torch.save(sampler_policy.policy.state_dict(), os.path.join(FLAGS.logging.output_dir, '{}_{}x{}_actor.pth'.format(FLAGS.env_list, FLAGS.unreal_dynamics, FLAGS.variety_list)))

        metrics['rollout_time'] = rollout_timer()
        metrics['train_time'] = train_timer()
        metrics['eval_time'] = eval_timer()
        metrics['epoch_time'] = rollout_timer() + train_timer() + eval_timer()
        wandb_logger.log(metrics)
        viskit_metrics.update(metrics)
        logger.record_dict(metrics)
        logger.dump_tabular(with_prefix=False, with_timestamp=False)

    if FLAGS.save_replaybuffer:
        data_dict = replay_buffer.sample_np()
        np.save("replay_buffer_2xgravity.npy", np.hstack((data_dict['observations'], data_dict['actions'], data_dict['rewards'], data_dict['next_observations'], data_dict['dones'])))

    if FLAGS.save_model:
        save_data = {'sac': sac, 'variant': variant, 'epoch': epoch}
        wandb_logger.save_pickle(save_data, '{}_{}x{}_model.pkl'.format(FLAGS.env_list, FLAGS.unreal_dynamics, FLAGS.variety_list))
        torch.save(sampler_policy.policy.state_dict(), os.path.join(FLAGS.logging.output_dir, '{}_{}x{}_actor.pth'.format(FLAGS.env_list, FLAGS.unreal_dynamics, FLAGS.variety_list)))


if __name__ == '__main__':
    absl.app.run(main)