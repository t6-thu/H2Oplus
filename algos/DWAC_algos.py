import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.serialization import load
import wandb
import time
import gym
import d4rl
import os
from torch.distributions import MultivariateNormal, Normal
from Network.Actor_Critic_net import Actor, Q_critic_distribution, V_critic, Q_critic, BC, Alpha, BC_VAE

ALPHA_MAX = 500


class HPO:
    max_grad_norm = 0.5
    soft_update = 0.01

    def __init__(self, env, num_state, num_action, num_hidden, replay_buffer,
                 lr=1e-4, gamma=0.99, warmup_steps=30000, alpha=100,
                 auto_alpha=False, epsilon=1, batch_size=256, device='cpu', A_from='Q', A_norm=False):
        super(HPO, self).__init__()
        self.learning_rate = lr
        self.device = device
        self.gamma = gamma
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.auto_alpha = auto_alpha
        self.alpha = alpha
        self.epsilon = epsilon
        self.cov_var = torch.full(size=(3,), fill_value=0.5).to(device)
        self.cov_mat = torch.diag(self.cov_var).to(device)
        self.actor_net = Actor(num_state, num_action, num_hidden, device).float().to(device)
        self.v_net = V_critic(num_state, num_hidden, device).float().to(device)
        self.bc_net = BC(num_state, num_action, num_hidden, device).float().to(device)
        self.bc_vae_net = BC_VAE(num_state, num_action, num_hidden, 4, device).float().to(device)
        self.q_net = Q_critic_distribution(num_state, num_action, num_hidden, device).float().to(device)
        self.target_q_net = Q_critic_distribution(num_state, num_action, num_hidden, device).float().to(device)
        self.alpha_net = Alpha(num_state, num_action, num_hidden, device).float().to(device)
        self.actor_optim = optim.Adam(self.actor_net.parameters(), self.learning_rate)
        self.v_optim = optim.Adam(self.v_net.parameters(), self.learning_rate)
        self.bc_optim = optim.Adam(self.bc_net.parameters(), self.learning_rate)
        self.bc_vae_optim = optim.Adam(self.bc_vae_net.parameters(), self.learning_rate)
        self.q_optim = optim.Adam(self.q_net.parameters(), self.learning_rate)
        self.alpha_optim = optim.Adam(self.alpha_net.parameters(), self.learning_rate)
        self.env = env
        self.replay_buffer = replay_buffer
        self.A_from = A_from
        self.A_norm = A_norm
        self.logger = {
            'delta_t': time.time_ns(),
            't_so_far': 0,  # time_steps so far
            'i_so_far': 0,  # iterations so far
            'batch_lens': [],  # episodic lengths in batch
            'batch_rews': [],  # episodic returns in batch
        }

    def learn_DWAC(self, total_time_step=1e+10):

        t_so_far = 0
        i_so_far = 0
        # ALG STEP 2

        if os.path.exists('hopper_medium_v2_bc_vae.pth'):
            self.load_parameters()
        else:
            self.pretrain_bc()
            self.pretrain_bc_vae()
            self.pretrain_Q()
            self.pretrain_V()

        while i_so_far < total_time_step:
            t_so_far += self.batch_size
            i_so_far += 1
            self.logger['t_so_far'] = t_so_far
            self.logger['i_so_far'] = i_so_far

            state, _, next_state, _, reward, not_done = self.replay_buffer.sample(self.batch_size)

            # sample action from \pi
            mean_pi = self.actor_net(state)
            distribution_pi = MultivariateNormal(mean_pi, self.cov_mat)
            action_pi = distribution_pi.rsample()
            action_pi = torch.clip(action_pi, -1, 1)

            # A = Q // A = r + V' -V // A = Q -V
            if self.A_from == 'Q':
                A = self.q_net(state, action_pi)
            elif self.A_from == 'V':
                A = self.get_A_from_V(state, next_state, reward, not_done)
            else:  # self.A_from == 'QV':
                A = self.get_A_from_QV(state, action_pi)

            if self.A_norm == True:
                A = (A - A.mean())/(A.std() + 1e-8)
            
            # log_pi = distribution_pi.log_prob(action_pi)
            # log_pi = torch.clip(log_pi, max=5)

            # conditional vae
            # z = torch.randn((self.batch_size, 4)).to(self.device)
            # action_mean, action_sigma = self.bc_vae_net.decode(z, state)
            # distribution_miu = Normal(action_mean, action_sigma)
            # log_miu = torch.mean(distribution_miu.log_prob(action_pi), dim=1).unsqueeze(1)

            # bc_multivariate
            mean_miu = self.bc_net(state)
            distribution_miu = MultivariateNormal(mean_miu, self.cov_mat)
            log_miu = distribution_miu.log_prob(action_pi).unsqueeze(1)
            # ratio = torch.exp(log_pi) - torch.exp(log_miu)

            if self.auto_alpha:
                self.update_alpha(state, action_pi.detach(), A.detach(), log_miu.detach())
                self.alpha = torch.mean(torch.clip(self.get_alpha(state, action_pi), max=ALPHA_MAX)).detach()
                wandb.log({"alpha":self.alpha.item()})

            wandb.log({"A_mean":A.mean().item()})
            actor_loss = -1. * torch.mean(A) #  + self.alpha * log_miu)
            self.actor_optim.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.max_grad_norm)
            self.actor_optim.step()
            wandb.log({"actor_loss": actor_loss})

            if i_so_far % 3000 == 0:
                self.rollout(pi='pi')
                # self.rollout(pi='miu')
                # self.rollout(pi='vae')
        # self._log_summary()

    def pretrain_bc_vae(self):
        for _ in range(self.warmup_steps):
            state, action, _, _, _, _ = self.replay_buffer.sample(2048)
            self.train_bc_vae(s=state, a=action)
        torch.save(self.bc_vae_net.state_dict(), "hopper_medium_v2_bc_vae.pth")

    def pretrain_V(self):
        for _ in range(self.warmup_steps):
            state, _, next_state, _, reward, not_done = self.replay_buffer.sample(2048)
            self.train_V(s=state, next_s=next_state, r=reward, not_done=not_done)
        torch.save(self.bc_vae_net.state_dict(), "hopper_medium_v2_v.pth")

    def pretrain_Q(self):
        for _ in range(self.warmup_steps):
            state, action, next_state, next_action, reward, not_done = self.replay_buffer.sample(2048)
            self.train_Q(s=state, a=action, next_s=next_state, next_a=next_action, r=reward, not_done=not_done)
        torch.save(self.bc_vae_net.state_dict(), "hopper_medium_v2_q.pth")
    
    def pretrain_bc(self):
        for _ in range(self.warmup_steps):
            state, action, _, _, _, _ = self.replay_buffer.sample(2048)
            self.train_bc(s=state, a=action)
        torch.save(self.bc_vae_net.state_dict(), "hopper_medium_v2_q.pth")           
    

    def update_alpha(self, state, action, A, log_miu):
        alpha = self.alpha_net(state, action)
        self.alpha_optim.zero_grad()
        alpha_loss = torch.mean(A + alpha * (log_miu + self.epsilon))
        alpha_loss.backward()
        self.alpha_optim.step()

    def get_alpha(self, state, action):
        return self.alpha_net(state, action).detach()

    def train_bc_vae(self, s, a):
        action_mean, action_sigma, mean, log_var = self.bc_vae_net(s, a)
        kl_div = - 0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        distribution = Normal(action_mean, action_sigma)
        action_log_prob = -1. * distribution.log_prob(a)
        loss_vae = torch.sum(action_log_prob)
        loss = loss_vae + kl_div
        self.bc_vae_optim.zero_grad()
        loss.backward()
        wandb.log({"bc_vae_loss": loss.item()})
        self.bc_vae_optim.step()

    def train_bc(self, s, a):
        mean = self.bc_net(s)
        distribution = MultivariateNormal(mean, self.cov_mat)
        action_log_prob = -1. * distribution.log_prob(a)
        loss_bc = torch.mean(action_log_prob)

        self.bc_optim.zero_grad()
        loss_bc.backward()
        self.bc_optim.step()
        wandb.log({"bc_loss": loss_bc.item()})

    def train_Q(self, s, a, next_s, next_a, r, not_done):
        target_q = r + torch.multiply(not_done, self.target_q_net(next_s, next_a).detach()) * self.gamma
        loss_q = nn.MSELoss()(target_q, self.q_net(s, a))

        self.q_optim.zero_grad()
        loss_q.backward()
        self.q_optim.step()
        wandb.log({"q_loss": loss_q.item()})

        # target Q update
        for target_param, param in zip(self.target_q_net.parameters(), self.q_net.parameters()):
            target_param.data.copy_(self.soft_update * param + (1 - self.soft_update) * target_param)

    def train_V(self, s, next_s, r, not_done):
        with torch.no_grad():
            target_v = r + torch.multiply(not_done, self.v_net(next_s)) * self.gamma
        loss_v = nn.MSELoss()(target_v, self.v_net(s))

        self.v_optim.zero_grad()
        loss_v.backward()
        self.v_optim.step()
        wandb.log({"v_loss": loss_v.item()})

    def get_A_from_QV(self, s, a):
        Q = self.q_net(s, a)
        V = self.v_net(s)
        return Q - V

    def get_A_from_V(self, s, next_s, r, not_done):
        with torch.no_grad():
            target_v = r + not_done * self.gamma * self.v_net(next_s)
            v = self.v_net(s)
        return target_v - v

    def choose_action_from_pi(self, s):
        mean = self.actor_net(s)
        distribution = MultivariateNormal(mean, self.cov_mat)
        action = distribution.sample()
        return action.cpu().detach().numpy()

    def choose_action_from_miu(self, s):
        mean = self.bc_net(s)
        distribution = MultivariateNormal(mean, self.cov_mat)
        action = distribution.sample()
        return action.cpu().detach().numpy()

    def choose_action_from_vae(self, s):
        z = torch.randn(4)
        z = torch.tensor(z, dtype=torch.float).unsqueeze(0).to(self.device)
        s = torch.tensor(s, dtype=torch.float).unsqueeze(0).to(self.device)
        action_mean, action_sigma = self.bc_vae_net.decode(z,s)
        distribution = Normal(action_mean, action_sigma)
        action = distribution.sample()
        return action.cpu().detach().numpy()

    def evaluate_pi(self, s, a):
        value = self.v_net(s).squeeze()
        mean = self.actor_net(s)
        dist = MultivariateNormal(mean, self.cov_mat)
        log_probs = dist.log_prob(a)
        entropy_loss = dist.entropy().mean()
        return value, log_probs, entropy_loss

    def evaluate_bc(self, s, a):
        mean = self.bc_net(s)
        dist = MultivariateNormal(mean, self.cov_mat)
        log_probs = dist.log_prob(a)
        return log_probs.detach()

    def rollout(self, pi='pi'):
        batch_obs = []
        batch_acts = []
        batch_log_probs = []
        batch_rews = []
        batch_lens = []
        for _ in range(5):
            ep_rews = []
            state = self.env.reset()
            ep_t = 0
            while True:
                ep_t += 1
                batch_obs.append(state)

                if pi == 'pi':
                    action = self.choose_action_from_pi(state)
                if pi == 'miu':
                    action = self.choose_action_from_miu(state)
                if pi == 'vae':
                    action = self.choose_action_from_vae(state)

                state, reward, done, _ = self.env.step(action)

                ep_rews.append(reward)

                # If the environment tells us the episode is terminated, break
                if done:
                    break

            # Track episodic lengths and rewards
            batch_lens.append(ep_t + 1)
            batch_rews.append(ep_rews)

        # Reshape data as tensors in the shape specified in function description, before returning
        batch_rtgs = self.compute_return(batch_rews).to(self.device)  # ALG STEP 4

        avg_ep_lens = np.mean(batch_lens)
        avg_ep_rews = np.mean([np.sum(ep_rews) for ep_rews in batch_rews])
        if pi == 'pi':
            wandb.log({"pi_episode_reward": avg_ep_rews.item()})
            # wandb.log({"pi_episode_lens": avg_ep_lens.item()})
        else:
            wandb.log({"miu_episode_reward": avg_ep_rews.item()})
            # wandb.log({"miu_episode_lens": avg_ep_lens.item()})
        self.logger['batch_rews'] = batch_rews
        self.logger['batch_lens'] = batch_lens

        return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens

    def compute_return(self, batch_rews):
        batch_rtgs = []

        # Iterate through each episode
        for ep_rews in reversed(batch_rews):
            discounted_reward = 0  # The discounted reward so far
            # Iterate through all rewards in the episode. We go backwards for smoother calculation of each
            # discounted return (think about why it would be harder starting from the beginning)
            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.gamma
                batch_rtgs.insert(0, discounted_reward)

        # Convert the rewards-to-go into a tensor
        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float).to(self.device)

        return batch_rtgs

    def _log_summary(self):
        # Calculate logging values. I use a few python shortcuts to calculate each value
        # without explaining since it's not too important to PPO; feel free to look it over,
        # and if you have any questions you can email me (look at bottom of README)
        delta_t = self.logger['delta_t']
        self.logger['delta_t'] = time.time_ns()
        delta_t = (self.logger['delta_t'] - delta_t) / 1e9
        delta_t = str(round(delta_t, 2))

        t_so_far = self.logger['t_so_far']
        i_so_far = self.logger['i_so_far']
        avg_ep_lens = np.mean(self.logger['batch_lens'])
        avg_ep_rews = np.mean([np.sum(ep_rews) for ep_rews in self.logger['batch_rews']])

        # Print logging statements
        print(flush=True)
        print(f"-------------------- Iteration #{i_so_far} --------------------", flush=True)
        print(f"Average Episodic Length: {avg_ep_lens}", flush=True)
        print(f"Average Episodic Return: {avg_ep_rews}", flush=True)
        print(f"Timesteps So Far: {t_so_far}", flush=True)
        print(f"Iteration took: {delta_t} secs", flush=True)
        print(f"------------------------------------------------------", flush=True)
        print(flush=True)

        # Reset batch-specific logging data
        self.logger['batch_lens'] = []
        self.logger['batch_rews'] = []

    def save_parameters(self):
        torch.save(self.bc_net.state_dict(), "hopper_medium_v2_bc.pth")
        torch.save(self.q_net.state_dict(), "hopper_medium_v2_q.pth")
        torch.save(self.v_net.state_dict(), "hopper_medium_v2_v.pth")
        torch.save(self.bc_vae_net.state_dict(), "hopper_medium_v2_bc_vae.pth")

    def load_parameters(self):
        self.bc_net.load_state_dict(torch.load("hopper_medium_v2_bc.pth"))
        self.q_net.load_state_dict(torch.load("hopper_medium_v2_q.pth"))
        self.v_net.load_state_dict(torch.load("hopper_medium_v2_v.pth"))
        self.bc_vae_net.load_state_dict(torch.load("hopper_medium_v2_bc_vae.pth"))
