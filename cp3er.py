import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils

from model.networks import CPActor as Actor
from model.networks import MoGCritic as Critic
from model.networks import Encoder

from torch.distributions import Normal, Categorical, MixtureSameFamily

class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0,
                              2 * self.pad + 1,
                              size=(n, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x,
                             grid,
                             padding_mode='zeros',
                             align_corners=False)

class CP3ERAgent:
    def __init__(self, obs_shape, action_shape, device, lr, feature_dim,
                 hidden_dim, critic_target_tau,
                 update_every_steps, use_tb, num_expl_steps, replay_ratio=1):
        self.device = device
        self.critic_target_tau = critic_target_tau
        self.update_every_steps = update_every_steps
        self.use_tb = use_tb

        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim

        self.replay_ratio = replay_ratio
        self.num_expl_steps = num_expl_steps

        # mog
        self.num_groups = None      # GroupNorm or LayerNorm
        self.num_components = 3     # Number of Gaussian 
        self.init_scale = 1e-3

        # models
        self.encoder, self.actor, self.critic, self.critic_target = self.init_models()
        self.critic_target.load_state_dict(self.critic.state_dict())

        # optimizers
        self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        # data augmentation
        self.aug = RandomShiftsAug(pad=4)

        self.train()
        self.critic_target.train()
    
    def init_models(self):
        encoder = Encoder(self.obs_shape).to(self.device)
        actor = Actor(encoder.repr_dim, self.action_shape[0], self.device, 
                      self.feature_dim, self.hidden_dim)
        # critic
        critic = Critic(encoder.repr_dim, self.action_shape[0], self.feature_dim,self.hidden_dim,
                             self.num_groups,self.num_components,self.init_scale).to(self.device)
        critic_target = Critic(encoder.repr_dim, self.action_shape[0], self.feature_dim,self.hidden_dim,
                             self.num_groups,self.num_components,self.init_scale).to(self.device)
        return encoder, actor, critic, critic_target
    
    def train(self, training=True):
        self.training = training
        self.encoder.train(training)
        self.actor.train(training)
        self.critic.train(training)
    
    # act without exp
    def act(self, obs):
        obs = torch.as_tensor(obs, device=self.device)
        obs = self.encoder(obs.unsqueeze(0))
        action = self.actor(obs)
        return action.cpu().numpy()[0]

    def to_distribution(self, mus, stdevs, logits):
        if self.num_components == 1:
            # For a single component, create a standard normal distribution
            dist = Normal(loc=mus[:, 0], scale=stdevs[:, 0])
        else:
            # For multiple components, create a mixture of Gaussian distributions
            dist = MixtureSameFamily(
                mixture_distribution=Categorical(logits=logits),
                component_distribution=Normal(loc=mus, scale=stdevs)
            )
        return dist

    def update_critic(self, obs, action, reward, discount, next_obs, step):
        metrics = dict()

        critic_loss , aux = self._compute_critic_loss(obs, action, reward, discount,next_obs, step)

        if self.use_tb:
            metrics['critic_q'] = aux['online_Q_mean']
            metrics['critic_target_q'] = aux['target_Q_mean']
            metrics['critic_loss_std'] = aux['critic_loss_std']
            metrics['critic_loss'] = aux['critic_loss']
   
        # optimize encoder and critic
        self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        self.encoder_opt.step()

        return metrics

    def update_actor(self, obs, action, step):
        metrics = dict()

        new_action = self.actor(obs)


        critic_info = self.critic(obs, new_action)
        mus, stdevs, logits = critic_info['mus'], critic_info['stdevs'], critic_info['logits']
        critic_dist = self.to_distribution(mus, stdevs, logits)

        q_estimate  = critic_dist.mean
        q_loss = -torch.mean(q_estimate)

        bc_loss = self.actor.cm.consistency_losses(action, obs)
        actor_loss = q_loss + 0.05 * bc_loss

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        if self.use_tb:
            metrics['actor_loss'] = actor_loss.item()
            metrics['q_loss'] = q_loss.item()
            metrics['bc_loss'] = bc_loss.item()

        return metrics

    def update(self, replay_iter, step):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_iter)
        org_obs, action, reward, discount, org_next_obs = utils.to_torch(
            batch, self.device)
        
        reward = reward.unsqueeze(1) if reward.dim() == 1 else reward
        discount = discount.unsqueeze(1) if discount.dim() == 1 else discount

        if self.use_tb:
            metrics['batch_reward'] = reward.mean().item()

        for _ in range(self.replay_ratio):
            # augment
            obs = self.aug(org_obs.float())
            next_obs = self.aug(org_next_obs.float())
            org_obs =  obs / 255.0 - 0.5
            obs = self.encoder(obs)

            with torch.no_grad():
                next_obs = self.encoder(next_obs)

            # update critic
            critic_metrics = self.update_critic(obs, action, reward, discount, next_obs, step)
            # update actor
            actor_metrics = self.update_actor(obs.detach(), action, step)
            
            # update critic target
            utils.soft_update_params(self.critic, self.critic_target,
                                    self.critic_target_tau)
            
        metrics.update(critic_metrics)
        metrics.update(actor_metrics)

        return metrics
    
    def _compute_critic_loss(self,  obs, act, rew, discount, next_obs, step):

        with torch.no_grad():

            next_action = self.actor(next_obs)

            target_info = self.critic_target(next_obs, next_action)
            mus, stdevs, logits = target_info['mus'], target_info['stdevs'], target_info['logits']

            if self.init_scale == 0:
                target_Q_dist = self.to_distribution(mus, stdevs, logits)
                target_Q = target_Q_dist.mean
            else:
                target_Q_dist = self.to_distribution(mus, stdevs, logits)
                target_Q = target_Q_dist.sample((20,))

            # compute target_Q
            target_Q = rew + discount * target_Q

        online_info = self.critic(obs, act)
        mus, stdevs, logits = online_info['mus'], online_info['stdevs'], online_info['logits']
        online_Q_dist = self.to_distribution(mus, stdevs, logits)

        # compute loss 
        critic_loss = -online_Q_dist.log_prob(target_Q)
        critic_loss_mean = critic_loss.mean()
        critic_loss_std = critic_loss.std()

        aux = {
            'critic_loss': critic_loss_mean.item(),
            'critic_loss_std': critic_loss_std.item(),
            'target_Q_mean': target_Q.mean().item(),
            'online_Q_mean': online_Q_dist.mean.mean().item(),
        }

        return critic_loss_mean, aux
