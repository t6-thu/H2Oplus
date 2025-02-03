import os
import time
import ipdb
import sys
from copy import deepcopy
import uuid
from tqdm import trange

import numpy as np
import pprint

import gym
import robel
import torch
import d4rl
import h5py
import pickle
import json
import wandb

import absl.app
import absl.flags

from conservative_sac import ConservativeSAC
from replay_buffer import batch_to_torch, get_d4rl_dataset, subsample_batch
from model import TanhGaussianPolicy, FullyConnectedQFunction, SamplerPolicy
from sampler import StepSampler, TrajSampler
from utils import Timer, WandBLogger, define_flags_with_default, set_random_seed, print_flags, get_user_flags, prefix_metrics

sys.path.append("..")

from viskit.logging import logger, setup_logger


FLAGS_DEF = define_flags_with_default(
    name_str='_',
    env='halfcheetah-medium-replay',
    max_traj_length=1000,
    seed=42,
    device='cuda',
    save_model=False,
    batch_size=256,

    reward_scale=1.0,
    reward_bias=0.0,
    clip_action=1,
    real_residual_ratio=1.,

    policy_arch='256-256',
    qf_arch='256-256',
    orthogonal_init=False,
    policy_log_std_multiplier=1.0,
    policy_log_std_offset=-1.0,

    n_epochs=1000,
    bc_epochs=0,
    n_train_step_per_epoch=1000,
    eval_period=10,
    eval_n_trajs=5,

    cql=ConservativeSAC.get_default_config(),
    logging=WandBLogger.get_default_config(),
)


def main(argv):
    FLAGS = absl.flags.FLAGS

    variant = get_user_flags(FLAGS, FLAGS_DEF)
    wandb_logger = WandBLogger(config=FLAGS.logging, variant=variant)
    wandb.run.name = f"{FLAGS.name_str}{FLAGS.real_residual_ratio}xdata_{FLAGS.env}_seed={FLAGS.seed}"
    
    setup_logger(
        variant=variant,
        exp_id=wandb_logger.experiment_id,
        seed=FLAGS.seed,
        base_log_dir=FLAGS.logging.output_dir,
        include_exp_prefix_sub_dir=False
    )

    set_random_seed(FLAGS.seed)

    # dataset = get_d4rl_dataset(eval_sampler.env)
    if FLAGS.env == "DKittyWalkRandom":
        folder_paths = ["../d4rl_mujoco_dataset/DKitty_replay_buffer_169398/target_1m", "../d4rl_mujoco_dataset/DKitty_replay_buffer_169398/target_2m"]
        all_json_data = []

        # Load data from DKitty Json files in both folders
        for folder_path in folder_paths:
            for filename in os.listdir(folder_path):
                if filename.endswith(".json"):
                    file_path = os.path.join(folder_path, filename)
                    with open(file_path, 'r') as json_file:
                        print(file_path)
                        json_data = json.load(json_file)

                    all_json_data.extend(json_data)
        
        # Flatten the JSON data
        # flat_all_json_data = [item for sublist in all_json_data for item in sublist]

        # Calculate the total number of data points
        total_num = len(all_json_data)

        # Generate random indices and sort them
        idx = np.sort(np.random.choice(range(total_num), int(total_num * FLAGS.real_residual_ratio), replace=False))

        data = {}
        # Load data based on the sorted indices
        data['observations'] = np.array([all_json_data[i]["state"] for i in idx]).astype(np.float32)
        data['actions'] = np.array([all_json_data[i]["action"] for i in idx]).astype(np.float32)
        data['rewards'] = np.array([all_json_data[i]["reward"] for i in idx]).astype(np.float32).reshape(-1, 1)
        data["next_observations"] = np.array([all_json_data[i]["next_state"] for i in idx]).astype(np.float32)
        data['terminals'] = np.array([all_json_data[i]["done"] for i in idx]).astype(np.bool).reshape(-1, 1)
        
        eval_sampler = TrajSampler(gym.make(FLAGS.env+'-v0').unwrapped, FLAGS.max_traj_length)
    elif FLAGS.env == "Humanoid":
        # with open("../d4rl_mujoco_dataset/{}-v2.pickle".format(FLAGS.env), "rb") as f:
        with open("../d4rl_mujoco_dataset/dataset_1934657_r6077.33_s0.00.pickle", "rb") as f:
            # dataset = pickle.load(f)
            try:
                # load complete file
                dataset = pickle.load(f)
            except pickle.UnpicklingError:
                # if loading fails, go back to beginning
                f.seek(0)
                # load data before truncated
                dataset = pickle.load(f)
            total_num = len(dataset)
            # ratio = 0.1  #* how much ratio of data to extract
            idx = sorted(np.random.choice(range(total_num), int(total_num * FLAGS.real_residual_ratio), replace=False))
            data = {}
            data['observations'] = np.array([dataset[i][0] for i in idx]).astype(np.float32) # An (N, dim_observation)-dimensional numpy array of observations
            data['actions'] = np.array([dataset[i][1] for i in idx]).astype(np.float32) # An (N, dim_action)-dimensional numpy array of actions
            data['rewards'] = np.expand_dims(np.array([dataset[i][2] for i in idx]).astype(np.float32), axis=1) # An (N,)-dimensional numpy array of rewards
            data["next_observations"] = np.array([dataset[i][3] for i in idx]).astype(np.float32) # An (N, dim_observation)-dimensional numpy array of next observations
            data['terminals'] = np.expand_dims(np.array([dataset[i][4] for i in idx]), axis=1) # An (N,)-dimensional numpy array of terminal flags
            # data['observations'] = dataset['observations'][idx, :]
            # data['actions'] = dataset['actions'][idx, :]
            # data['next_observations'] = dataset['next_observations'][idx, :]
            # data['rewards'] = dataset['rewards'][idx]
            # data['terminals'] = dataset['terminals'][idx]
            data['rewards'] = data['rewards'] * FLAGS.reward_scale + FLAGS.reward_bias
            data['actions'] = np.clip(data['actions'], -FLAGS.clip_action, FLAGS.clip_action)
            
        eval_sampler = TrajSampler(gym.make(FLAGS.env+'-v2').unwrapped, FLAGS.max_traj_length)
    else:
        dataset = h5py.File("../d4rl_mujoco_dataset/{}-v2.hdf5".format(FLAGS.env.replace('-',"_")),"r")
        total_num = dataset['observations'].shape[0]
        idx = sorted(np.random.choice(range(total_num), int(total_num * FLAGS.real_residual_ratio), replace=False))
        data = {}
        data['observations'] = dataset['observations'][idx, :]
        data['actions'] = dataset['actions'][idx, :]
        data['next_observations'] = dataset['next_observations'][idx, :]
        data['rewards'] = dataset['rewards'][idx]
        data['terminals'] = dataset['terminals'][idx]
        

        data['rewards'] = data['rewards'] * FLAGS.reward_scale + FLAGS.reward_bias
        data['actions'] = np.clip(data['actions'], -FLAGS.clip_action, FLAGS.clip_action)
        
        eval_sampler = TrajSampler(gym.make(FLAGS.env+'-v2').unwrapped, FLAGS.max_traj_length)

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

    if FLAGS.cql.target_entropy >= 0.0:
        FLAGS.cql.target_entropy = -np.prod(eval_sampler.env.action_space.shape).item()

    sac = ConservativeSAC(FLAGS.cql, policy, qf1, qf2, target_qf1, target_qf2)
    sac.torch_to_device(FLAGS.device)

    sampler_policy = SamplerPolicy(policy, FLAGS.device)

    viskit_metrics = {}
    for epoch in trange(FLAGS.n_epochs):
        metrics = {'epoch': epoch}

        with Timer() as train_timer:
            for batch_idx in trange(FLAGS.n_train_step_per_epoch):
                batch = subsample_batch(data, FLAGS.batch_size)
                batch = batch_to_torch(batch, FLAGS.device)
                metrics.update(prefix_metrics(sac.train(batch, bc=epoch < FLAGS.bc_epochs), 'sac'))

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
                    wandb_logger.save_pickle(save_data, 'model.pkl')

        metrics['train_time'] = train_timer()
        metrics['eval_time'] = eval_timer()
        metrics['epoch_time'] = train_timer() + eval_timer()
        wandb_logger.log(metrics)
        viskit_metrics.update(metrics)
        logger.record_dict(viskit_metrics)
        logger.dump_tabular(with_prefix=False, with_timestamp=False)

    if FLAGS.save_model:
        save_data = {'sac': sac, 'variant': variant, 'epoch': epoch}
        wandb_logger.save_pickle(save_data, 'model.pkl')

if __name__ == '__main__':
    absl.app.run(main)
