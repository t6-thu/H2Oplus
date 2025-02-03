import datetime
import sys
sys.path.append("/home/lbq/.local/share/ov/pkg/isaac_sim-2022.1.1/When-to-trust-your-simulator")
sys.path.append('../')
from copy import deepcopy
import absl.app
import absl.flags
import numpy as np
import torch
import wandb
from tqdm import trange
from SimpleSAC.envs import get_isaac_env
def env_test():
    env_name: str = 'WheelLegged'
    backend = 'torch'
    sim_params = {'use_gpu_pipeline': True}
    task_args = {'device': 'cuda'}
    env = get_isaac_env(
        env_name=env_name,
        backend=backend,
        sim_params=sim_params,
        task_args=task_args,
    )
    print(env.step(torch.ones([1])))
    env.close()


from SimpleSAC.model import FullyConnectedQFunction, FullyConnectedNetwork, SamplerPolicy, TanhGaussianPolicy
from SimpleSAC.sampler import StepSampler, TrajSampler
from SimpleSAC.drh2o import H2OPLUS
from sac import SAC
from SimpleSAC.utils import (Timer, WandBLogger, define_flags_with_default,
                             get_user_flags, prefix_metrics, print_flags,
                             set_random_seed)

from Network.Dynamics_net import Dynamics
from Network.Weight_net import ConcatDiscriminator, ConcatRatioEstimator
from viskit.logging import logger, setup_logger
from SimpleSAC.mixed_replay_buffer import NewReplayBuffer
from SimpleSAC.replay_buffer import ReplayBuffer, batch_to_torch

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
    max_traj_length=1000,
    replay_buffer_size=1000000, #extra
    seed=42,
    device='cuda',
    save_model=True,
    batch_size=32,

    reward_scale=1.0,
    reward_bias=0.0,
    clip_action=1.0,
    joint_noise_std=0.0,

    policy_arch='32-32',
    qf_arch='32-32',
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
    eval_period=50,
    eval_n_trajs=5,

    # h2o=H2OPLUS.get_default_config(),
    sac=SAC.get_default_config(),

    logging=WandBLogger.get_default_config(),
    # isaac env
    env_name='WheelLegged',
    file_path='../data/5.11.s',
    backend='torch',
    sim_params_use_gpu_pipeline=True,
    task_args_device='cuda',
)


def main(argv):
    FLAGS = absl.flags.FLAGS
    FLAGS.logging.entity = None
    FLAGS.logging.online = True
    FLAGS.logging.output_dir = f'./experiment_output/standing-n-sac'
    # FLAGS.h2o.quantile = 0.9
    # FLAGS.h2o.backup_policy_entropy = True
    # FLAGS.h2o.batch_sim_ratio = 0.5
    # FLAGS.h2o.exploit_coeff = 0.1

    # define logged variables for wandb
    variant = get_user_flags(FLAGS, FLAGS_DEF)
    wandb_logger = WandBLogger(config=FLAGS.logging, variant=variant)
    wandb.run.name = f"SAC_{FLAGS.env_name}_{FLAGS.file_path.split('/')[-1]}_seed={FLAGS.seed}_learnedDynamics={FLAGS.dynamics_model}_{FLAGS.current_time}"

    setup_logger(
        variant=variant,
        exp_id=wandb_logger.experiment_id,
        seed=FLAGS.seed,
        base_log_dir=FLAGS.logging.output_dir,
        include_exp_prefix_sub_dir=False
    )

    set_random_seed(FLAGS.seed)

    real_env = get_isaac_env(
        FLAGS.env_name,
        backend=FLAGS.backend,
        sim_params={'use_gpu_pipeline': FLAGS.sim_params_use_gpu_pipeline},
        task_args={'device': FLAGS.task_args_device},
    )

    # a step sampler for "simulated" training
    train_sampler = StepSampler(real_env, FLAGS.max_traj_length)
    # a trajectory sampler for "real-world" evaluation
    eval_sampler = TrajSampler(real_env, FLAGS.max_traj_length)

    # replay buffer
    num_state = real_env.observation_space.shape[0]
    num_action = real_env.action_space.shape[0]
    replay_buffer = ReplayBuffer(num_state, num_action, FLAGS.replay_buffer_size)
    '''
    replay_buffer = NewReplayBuffer(
        file_path=FLAGS.file_path,
        env_name=FLAGS.env_name,
        reward_scale=FLAGS.reward_scale,
        reward_bias=FLAGS.reward_bias,
        clip_action=FLAGS.clip_action,
        state_dim=num_state,
        action_dim=num_action)
    '''
    # Should a dynamics model be learned for s' sampling when estimating u(s,a)?
    if FLAGS.dynamics_model:
        # initialize dynamics model
        dynamics_model = Dynamics(num_state, num_action, 256, dropout=FLAGS.model_dropout, device=FLAGS.device).to(
            FLAGS.device)
        model_optimizer = torch.optim.Adam(dynamics_model.parameters(), lr=FLAGS.model_lr)
        for n in trange(FLAGS.model_train_epoch):
            real_obs, real_action, real_next_obs = replay_buffer.sample(FLAGS.batch_size, scope="real",
                                                                        type="sas").values()
            minus_logp_pi = dynamics_model.get_loss(real_obs, real_action, real_next_obs - real_obs)
            model_optimizer.zero_grad()
            minus_logp_pi.backward()
            model_optimizer.step()
            if n % 100 == 0:
                metrics = {}
                metrics['model_loss'] = minus_logp_pi.cpu().detach().numpy().item()
                wandb_logger.log(metrics)
        xi_sas = ConcatRatioEstimator(2 * num_state + num_action, 256, 1, FLAGS.device, scale=FLAGS.tanh_scale,
                                      dropout=FLAGS.dis_dropout).float().to(FLAGS.device)
    else:
        dynamics_model = None

    # discirminators
    d_sa = ConcatDiscriminator(num_state + num_action, 256, 2, FLAGS.device, dropout=FLAGS.dis_dropout).float().to(
        FLAGS.device)
    d_sas = ConcatDiscriminator(2 * num_state + num_action, 256, 2, FLAGS.device, dropout=FLAGS.dis_dropout).float().to(
        FLAGS.device)

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

    # if FLAGS.h2o.target_entropy >= 0.0:
        # FLAGS.h2o.target_entropy = -np.prod(eval_sampler.env.action_space.shape).item()

    if FLAGS.sac.target_entropy >= 0.0:
        FLAGS.sac.target_entropy = -np.prod(eval_sampler.env.action_space.shape).item()

    '''
    if FLAGS.dynamics_model:
        h2o = H2OPLUS(FLAGS.h2o, policy, qf1, qf2, target_qf1, target_qf2, vf, replay_buffer,
                      dynamics_model=dynamics_model, dynamics_ratio_estimator=xi_sas)
    else:
        h2o = H2OPLUS(FLAGS.h2o, policy, qf1, qf2, target_qf1, target_qf2, vf, replay_buffer, d_sa=d_sa, d_sas=d_sas,
                      dynamics_model=dynamics_model)
    h2o.torch_to_device(FLAGS.device)
    '''

    sac = SAC(FLAGS.sac, policy, qf1, qf2, target_qf1, target_qf2)
    sac.torch_to_device(FLAGS.device)

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

                batch = replay_buffer.sample(FLAGS.batch_size)
                if batch_idx + 1 == FLAGS.n_train_step_per_epoch:
                    metrics.update(
                        prefix_metrics(sac.train(batch), 'sac')
                    )
                else:
                    sac.train(batch)
                '''  
                if batch_idx + 1 == FLAGS.n_train_step_per_epoch:
                    metrics.update(
                        prefix_metrics(h2o.train(FLAGS.batch_size), 'h2o')
                    )
                else:
                    h2o.train(FLAGS.batch_size)
                '''

        # TODO Evaluate in the real world
        with Timer() as eval_timer:
            if epoch == 0 or (epoch + 1) % FLAGS.eval_period == 0:
                trajs = eval_sampler.sample(
                    sampler_policy, FLAGS.eval_n_trajs, deterministic=True
                )
                '''
                if not FLAGS.dynamics_model:
                    eval_dsa_loss, eval_dsas_loss = h2o.discriminator_evaluate()
                    metrics['eval_dsa_loss'] = eval_dsa_loss
                    metrics['eval_dsas_loss'] = eval_dsas_loss
                ''' 
                metrics['average_return'] = np.mean([np.sum(t['rewards']) for t in trajs])
                metrics['average_traj_length'] = np.mean([len(t['rewards']) for t in trajs])
                metrics['average_normalizd_return'] = np.mean(
                    [np.sum(t['rewards']) for t in trajs]
                )

                if FLAGS.save_model:
                    save_data = {'sac': sac, 'variant': variant, 'epoch': epoch}
                    wandb_logger.save_pickle(save_data, f'model-{epoch}.pkl')
            '''
            if epoch == 0 or (epoch + 1) % FLAGS.eval_period == 0:
                trajs = eval_sampler.sample(
                    sampler_policy, FLAGS.eval_n_trajs, deterministic=True
                )

                metrics['average_return'] = np.mean([np.sum(t['rewards']) for t in trajs])
                metrics['average_traj_length'] = np.mean([len(t['rewards']) for t in trajs])
                metrics['average_normalizd_return'] = np.mean(
                    [eval_sampler.env.get_normalized_score(np.sum(t['rewards'])) for t in trajs]
                )

                if FLAGS.save_model:
                    save_data = {'sac': sac, 'variant': variant, 'epoch': epoch}
                    wandb_logger.save_pickle(save_data, 'model.pkl')
            '''
        metrics['rollout_time'] = rollout_timer()
        metrics['train_time'] = train_timer()
        metrics['eval_time'] = eval_timer()
        metrics['epoch_time'] = rollout_timer() + train_timer() + eval_timer()
        wandb_logger.log(metrics)
        viskit_metrics.update(metrics)
        logger.record_dict(viskit_metrics)
        logger.dump_tabular(with_prefix=False, with_timestamp=False)
    '''
    if FLAGS.save_model:
        save_data = {'h2o': h2o, 'variant': variant, 'epoch': epoch}
        wandb_logger.save_pickle(save_data, 'model.pkl')
    '''
    if FLAGS.save_model:
        save_data = {'sac': sac, 'variant': variant, 'epoch': epoch}
        wandb_logger.save_pickle(save_data, 'model.pkl')

    real_env.close()


if __name__ == '__main__':
    absl.app.run(main)
