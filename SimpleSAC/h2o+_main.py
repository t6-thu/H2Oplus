import datetime
import os
import pprint
import re
import sys
import time
import uuid
from copy import deepcopy
from sre_parse import FLAGS

import absl.app
import absl.flags
import d4rl
import gym
import robel
import numpy as np
import torch
import wandb
import ipdb
from tqdm import trange

from envs import get_new_density_env, get_new_friction_env, get_new_gravity_env, get_new_thigh_range_env, get_new_foot_shape_env, get_new_foot_stiffness_env, get_new_thigh_size_env, get_new_ellipsoid_limb_env, get_new_box_limb_env, get_new_head_size_env, get_new_torso_length_env, get_new_limb_stiffness_env, get_new_tendon_elasticity_env
from mixed_replay_buffer import MixedReplayBuffer
from model import FullyConnectedQFunction, FullyConnectedNetwork, SamplerPolicy, TanhGaussianPolicy
from sampler import StepSampler, TrajSampler
from h2oplus import H2OPLUS
from utils import (Timer, WandBLogger, define_flags_with_default,
                get_user_flags, prefix_metrics, print_flags,
                set_random_seed)

sys.path.append("..")

from Network.Dynamics_net import Dynamics
from Network.Weight_net import ConcatDiscriminator, ConcatRatioEstimator
from viskit.logging import logger, setup_logger

nowTime = datetime.datetime.now().strftime('%y-%m-%d-%H-%M-%S')

FLAGS_DEF = define_flags_with_default(
    current_time=nowTime,
    name_str='',
    env_list='HalfCheetah-v2',
    data_source='medium_replay',
    unreal_dynamics="gravity",
    variety_list="2.0",
    replaybuffer_ratio=10.,
    real_residual_ratio=1.,
    tanh_scale=2,
    dis_dropout=False,
    warmup_steps=0,
    max_traj_length=1000,
    seed=42,
    device='cuda',
    save_model=False,
    batch_size=256,

    reward_scale=1.0,
    reward_bias=0.0,
    clip_action=1.0,
    joint_noise_std=0.0,

    policy_arch='256-256',
    qf_arch='256-256',
    orthogonal_init=False,
    policy_log_std_multiplier=1.0,
    policy_log_std_offset=-1.0,

    # dynamics model
    dynamics_model=False,
    model_train_epoch=10000,
    model_lr=3e-4,
    model_dropout=False,

    # train and evaluate policy
    n_epochs=1000,
    bc_epochs=0,
    n_rollout_steps_per_epoch=1000,
    n_train_step_per_epoch=1000,
    eval_period=10,
    eval_n_trajs=5,

    h2o=H2OPLUS.get_default_config(),
    logging=WandBLogger.get_default_config()
)


def main(argv):
    FLAGS = absl.flags.FLAGS

    # define logged variables for wandb
    variant = get_user_flags(FLAGS, FLAGS_DEF)
    wandb_logger = WandBLogger(config=FLAGS.logging, variant=variant)
    wandb.run.name = f"{FLAGS.name_str}_{FLAGS.env_list}_{FLAGS.data_source}_{FLAGS.unreal_dynamics}x{FLAGS.variety_list}_seed={FLAGS.seed}_learnedDynamics={FLAGS.dynamics_model}_{FLAGS.current_time}"

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
            # different varieties: 0.5, 1.5, 2.0, ...
            for variety_degree in FLAGS.variety_list.split(";"):
                variety_degree = float(variety_degree)

                if env_name in ["DKittyWalkFixed-v0", "DKittyWalkRandom-v0", "DKittyWalkRandomDynamics-v0"]:
                    real_env = gym.make(env_name) # DKittyWalkFixed-v0, DKittyWalkRandom-v0, DKittyWalkRandomDynamics-v0
                    sim_env = gym.make("DKittyWalkRandomDynamics-v0")
                else:
                    if env_name == "Humanoid-v2":
                        off_env_name = env_name
                    else:
                        off_env_name = "{}-{}-v2".format(env_name.split("-")[0].lower(), FLAGS.data_source).replace('_',"-")
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

    # replay buffer
    num_state = real_env.observation_space.shape[0]
    num_action = real_env.action_space.shape[0]
    replay_buffer = MixedReplayBuffer(FLAGS.reward_scale, FLAGS.reward_bias, FLAGS.clip_action, num_state, num_action, task=env_name.split("-")[0].lower(), data_source=FLAGS.data_source, device=FLAGS.device, buffer_ratio=FLAGS.replaybuffer_ratio, residual_ratio=FLAGS.real_residual_ratio, max_episode_steps=real_env._max_episode_steps)
    # ipdb.set_trace()

    # Should a dynamics model be learned for s' sampling when estimating u(s,a)?
    if FLAGS.dynamics_model:
        # initialize dynamics model
        dynamics_model = Dynamics(num_state, num_action, 256, dropout=FLAGS.model_dropout, device=FLAGS.device).to(FLAGS.device)
        model_optimizer = torch.optim.Adam(dynamics_model.parameters(), lr=FLAGS.model_lr)
        for n in trange(FLAGS.model_train_epoch):
            real_obs, real_action, real_next_obs = replay_buffer.sample(FLAGS.batch_size, scope="real", type="sas").values()
            minus_logp_pi = dynamics_model.get_loss(real_obs, real_action, real_next_obs - real_obs)
            model_optimizer.zero_grad()
            minus_logp_pi.backward()
            model_optimizer.step()
            if n % 100 == 0:
                metrics = {}
                metrics['model_loss'] = minus_logp_pi.cpu().detach().numpy().item()
                wandb_logger.log(metrics)
        xi_sas = ConcatRatioEstimator(2 * num_state + num_action, 256, 1, FLAGS.device, scale=FLAGS.tanh_scale, dropout=FLAGS.dis_dropout).float().to(FLAGS.device) 
    else:
        dynamics_model = None


    # discirminators
    d_sa = ConcatDiscriminator(num_state + num_action, 256, 2, FLAGS.device, dropout=FLAGS.dis_dropout).float().to(FLAGS.device)
    d_sas = ConcatDiscriminator(2 * num_state + num_action, 256, 2, FLAGS.device, dropout=FLAGS.dis_dropout).float().to(FLAGS.device) 

    # agent
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
    
    vf = FullyConnectedNetwork(
        eval_sampler.env.observation_space.shape[0], 1,
        arch=FLAGS.qf_arch,
        orthogonal_init=FLAGS.orthogonal_init,
    )

    if FLAGS.h2o.target_entropy >= 0.0:
        FLAGS.h2o.target_entropy = -np.prod(eval_sampler.env.action_space.shape).item()

    if FLAGS.dynamics_model:
        h2o = H2OPLUS(FLAGS.h2o, policy, qf1, qf2, target_qf1, target_qf2, vf, replay_buffer, dynamics_model=dynamics_model, dynamics_ratio_estimator=xi_sas)
    else:
        h2o = H2OPLUS(FLAGS.h2o, policy, qf1, qf2, target_qf1, target_qf2, vf, replay_buffer, d_sa=d_sa, d_sas=d_sas, dynamics_model=dynamics_model)
    h2o.torch_to_device(FLAGS.device)

    # sampling policy is always the current policy: \pi
    sampler_policy = SamplerPolicy(policy, FLAGS.device)

    viskit_metrics = {}

    # train and evaluate for n_epochs
    for epoch in trange(FLAGS.n_epochs):
        metrics = {}

        # TODO rollout from the simulator
        with Timer() as rollout_timer:
            # rollout and append simulated trajectories to the replay buffer
            train_sampler.sample(
                sampler_policy, FLAGS.n_rollout_steps_per_epoch,
                deterministic=False, replay_buffer=replay_buffer, joint_noise_std=FLAGS.joint_noise_std
            )
            metrics['epoch'] = epoch

        # TODO Train from the mixed data
        with Timer() as train_timer:
            for batch_idx in trange(FLAGS.n_train_step_per_epoch):
                # batch = subsample_batch(dataset, FLAGS.batch_size)
                # batch = batch_to_torch(batch, FLAGS.device)
                # metrics.update(prefix_metrics(h2o.train(batch, bc=epoch < FLAGS.bc_epochs), 'h2o'))

                # real_batch_size = int(FLAGS.batch_size * (1 - FLAGS.batch_sim_ratio))
                # sim_batch_size = int(FLAGS.batch_size * FLAGS.batch_sim_ratio)
                # real_batch = replay_buffer.sample(FLAGS.batch_size * (1 - FLAGS.batch_sim_ratio), scope="real")
                # sim_batch = replay_buffer.sample(FLAGS.batch_size * FLAGS.batch_sim_ratio, scope="sim")
                # batch = [real_batch, sim_batch]
                if batch_idx + 1 == FLAGS.n_train_step_per_epoch:
                    metrics.update(
                        prefix_metrics(h2o.train(FLAGS.batch_size, FLAGS.warmup_steps), 'h2o')
                    )
                else:
                    h2o.train(FLAGS.batch_size, FLAGS.warmup_steps)

        # TODO Evaluate in the real world
        with Timer() as eval_timer:
            if epoch == 0 or (epoch + 1) % FLAGS.eval_period == 0:
                trajs = eval_sampler.sample(
                    sampler_policy, FLAGS.eval_n_trajs, deterministic=True
                )
                if not FLAGS.dynamics_model:
                    eval_dsa_loss, eval_dsas_loss = h2o.discriminator_evaluate()
                    metrics['eval_dsa_loss'] = eval_dsa_loss
                    metrics['eval_dsas_loss'] = eval_dsas_loss
                metrics['average_return'] = np.mean([np.sum(t['rewards']) for t in trajs])
                metrics['average_traj_length'] = np.mean([len(t['rewards']) for t in trajs])
                # metrics['average_normalizd_return'] = np.mean(
                #     [eval_sampler.env.get_normalized_score(np.sum(t['rewards'])) for t in trajs]
                # )
                
                if FLAGS.save_model:
                    save_data = {'h2o': h2o, 'variant': variant, 'epoch': epoch}
                    wandb_logger.save_pickle(save_data, 'model_{}.pkl'.format(epoch))

        metrics['rollout_time'] = rollout_timer()
        metrics['train_time'] = train_timer()
        metrics['eval_time'] = eval_timer()
        metrics['epoch_time'] = rollout_timer() + train_timer() + eval_timer()
        wandb_logger.log(metrics)
        viskit_metrics.update(metrics)
        logger.record_dict(viskit_metrics)
        logger.dump_tabular(with_prefix=False, with_timestamp=False)

    if FLAGS.save_model:
        save_data = {'h2o': h2o, 'variant': variant, 'epoch': epoch}
        wandb_logger.save_pickle(save_data, 'model.pkl')

if __name__ == '__main__':
    absl.app.run(main)
