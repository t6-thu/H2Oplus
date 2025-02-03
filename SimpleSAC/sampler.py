import numpy as np
import torch
import torch.nn.functional as F
# import ipdb


class StepSampler(object):

    def __init__(self, env, max_traj_length=1000, dis=None, device="cuda"):
        self.max_traj_length = max_traj_length
        self._env = env
        self._traj_steps = 0
        self._dis = dis
        self.device = device
        if self._dis:
            self.d_sa = dis[0]
            self.d_sas = dis[1]
            self.clip_dynamics_ratio_min = dis[2]
            self.clip_dynamics_ratio_max = dis[3]
        self._current_observation = self.env.reset()

    def sample(self, policy, n_steps, deterministic=False, replay_buffer=None, joint_noise_std=0.):
        observations = []
        actions = []
        rewards = []
        next_observations = []
        dones = []

        for _ in range(n_steps):
            self._traj_steps += 1
            observation = self._current_observation
            if isinstance(observation, torch.Tensor):
                observation = observation.cpu().numpy()

            #TODO sample actions from current policy
            action = policy(
                np.expand_dims(observation, 0), deterministic=deterministic
            )[0, :]
            # if joint_noise_std > 0.:
            #     # pdb.set_trace()
            #     action += np.random.randn(action.shape[0],) * joint_noise_std
            if joint_noise_std > 0.:
                # normal distribution
                next_observation, reward, done, _ = self.env.step(action + np.random.randn(action.shape[0],) * joint_noise_std)

                # # truncated normal distribution
                # next_observation, reward, done, _ = self.env.step(action + torch.nn.init.trunc_normal_(torch.zeros((action.shape)), a=0, b=1).numpy() * joint_noise_std)
                # # np.random.randn(action.shape[0],) * joint_noise_std)

                # # action[-1] noise
                # action[-1] += np.random.randn() * joint_noise_std
                # next_observation, reward, done, _ = self.env.step(action)
                # # np.random.randn(action.shape[0],) * joint_noise_std)

            else:
                next_observation, reward, done, _ = self.env.step(action)

            # if self._dis:
            #     sim_real_dynamics_ratio = self.sim_real_dynamics_ratio(observation, action, next_observation)
            #     ipdb.set_trace()
            #     if torch.logical_or(sim_real_dynamics_ratio >= self.clip_dynamics_ratio_max, sim_real_dynamics_ratio <= self.clip_dynamics_ratio_min):
            #         continue

            observations.append(observation)
            actions.append(action)
            if isinstance(next_observation, torch.Tensor):
                next_observation = next_observation.cpu().numpy()
            rewards.append(reward)
            dones.append(done)
            next_observations.append(next_observation)
            
            # if self._traj_steps == 1:
            #     replay_buffer.append_init_obs(observation)

            # # add samples derived from current policy to replay buffer
            # if replay_buffer is not None:
            #     replay_buffer.append(
            #         observation, action, reward, next_observation, done
            #     )

            self._current_observation = next_observation

            if done or self._traj_steps >= self.max_traj_length:
                self._traj_steps = 0
                self._current_observation = self.env.reset()

        if self._dis:
            sim_real_dynamics_ratio = self.sim_real_dynamics_ratio(observations, actions, next_observations)
            in_dynamics = (sim_real_dynamics_ratio < self.clip_dynamics_ratio_max) & (sim_real_dynamics_ratio > self.clip_dynamics_ratio_min)
            in_dynamics_index = [i for i, x in enumerate(in_dynamics) if x]
            observations = [observations[i] for i in range(len(observations)) if i in in_dynamics_index]
            actions = [actions[i] for i in range(len(actions)) if i in in_dynamics_index]
            rewards = [rewards[i] for i in range(len(rewards)) if i in in_dynamics_index]
            next_observations = [next_observations[i] for i in range(len(next_observations)) if i in in_dynamics_index]
            dones = [dones[i] for i in range(len(dones)) if i in in_dynamics_index]
            
        # add samples derived from current policy to replay buffer
        if replay_buffer is not None:
            replay_buffer.append_traj(
                observations, actions, rewards, next_observations, dones
            )
        
        return dict(
            observations=np.array(observations, dtype=np.float32),
            actions=np.array(actions, dtype=np.float32),
            rewards=np.array(rewards, dtype=np.float32),
            next_observations=np.array(next_observations, dtype=np.float32),
            dones=np.array(dones, dtype=np.float32),
        )

    @property
    def env(self):
        return self._env
    
    def sim_real_dynamics_ratio(self, observations, actions, next_observations):
        observations = torch.FloatTensor(observations).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        next_observations = torch.FloatTensor(next_observations).to(self.device)
        
        sa_logits = self.d_sa(observations, actions)
        sa_prob = F.softmax(sa_logits, dim=1)
        adv_logits = self.d_sas(observations, actions, next_observations)
        sas_prob = F.softmax(adv_logits + sa_logits, dim=1)

        with torch.no_grad():
            ratio = torch.clamp((sas_prob[:, 1] * sa_prob[:, 0]) / (sas_prob[:, 0] * sa_prob[:, 1]), min=self.clip_dynamics_ratio_min, max=self.clip_dynamics_ratio_max)
        # sa_logits = self.d_sa(observations, actions).cpu().detach().numpy()
        # sa_prob = np.exp(sa_logits) / np.sum(np.exp(sa_logits), dim=1)
        
        # adv_logits = self.d_sas(observations, actions, next_observations).cpu().detach().numpy()
        # sas_prob = np.exp(adv_logits + sa_logits) / np.sum(np.exp(adv_logits + sa_logits), dim=1)

        # ratio = np.clip((sas_prob[:, 1] * sa_prob[:, 0]) / (sas_prob[:, 0] * sa_prob[:, 1]), min=self.clip_dynamics_ratio_min, max=self.clip_dynamics_ratio_max)

        return ratio

# with dones as a trajectory end indicator, we can use this sampler to sample trajectories
class TrajSampler(object):

    def __init__(self, env, max_traj_length=1000):
        self.max_traj_length = max_traj_length
        self._env = env

    def sample(self, policy, n_trajs, deterministic=False, replay_buffer=None):
        trajs = []
        for _ in range(n_trajs):
            observations = []
            actions = []
            rewards = []
            next_observations = []
            dones = []

            observation = self.env.reset()
            if isinstance(observation, torch.Tensor):
                observation = observation.cpu().numpy()

            for _ in range(self.max_traj_length):
                action = policy(
                    np.expand_dims(observation, 0), deterministic=deterministic
                )[0, :]
                next_observation, reward, done, _ = self.env.step(action)
                if isinstance(next_observation, torch.Tensor):
                    next_observation = next_observation.cpu().numpy()

                observations.append(observation)
                actions.append(action)
                rewards.append(reward)
                dones.append(done)
                next_observations.append(next_observation)

                if replay_buffer is not None:
                    replay_buffer.add_sample(
                        observation, action, reward, next_observation, done
                    )

                observation = next_observation

                if done:
                    break

            trajs.append(dict(
                observations=np.array(observations, dtype=np.float32),
                actions=np.array(actions, dtype=np.float32),
                rewards=np.array(rewards, dtype=np.float32),
                next_observations=np.array(next_observations, dtype=np.float32),
                dones=np.array(dones, dtype=np.float32),
            ))

        return trajs

    @property
    def env(self):
        return self._env
