import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import utils
import algorithms.modules as m
import wandb

class SAC(object):
    def __init__(self, obs_shape, state_shape,  action_shape, args):

        obs_shape = (3, obs_shape[1], obs_shape[2])
        self.args = args
        self.tau = args.tau
        self.discount = args.discount
        self.critic_tau = args.critic_tau
        self.encoder_tau = args.encoder_tau
        self.actor_update_freq = args.actor_update_freq
        self.critic_target_update_freq = args.critic_target_update_freq
        self.from_state = args.observation_type=='state'

        self.use_prioritized_buffer = args.use_prioritized_buffer

        if args.observation_type != "state+image":
            state_shape = None
        self.state_shape = state_shape

        if self.from_state:
            shared = m.Identity(obs_shape)
            head = m.Identity(obs_shape)
        else:
            shared = m.SharedCNN(obs_shape, args.num_shared_layers, args.num_filters, args.mean_zero).cuda()
            head = m.HeadCNN(shared.out_shape, args.num_head_layers, args.num_filters).cuda()
       
        self.encoder_rl = m.Encoder(
            shared,
            head,
            m.Flatten(),
        ).cuda()
        self.encoder_rl.out_dim = head.out_shape[0]

        # self.actor = m.Actor(actor_encoder, action_shape, args.hidden_dim, args.actor_log_std_min, args.actor_log_std_max).cuda()
        # self.critic = m.Critic(critic_encoder, action_shape, args.hidden_dim).cuda()
        # self.critic_target = deepcopy(self.critic)

        self.actor = m.EfficientActor(self.encoder_rl.out_dim, args.projection_dim, action_shape, args.hidden_dim,
                                      args.actor_log_std_min, args.actor_log_std_max, state_shape, args.hidden_dim_state).cuda()
        self.critic = m.EfficientCritic(self.encoder_rl.out_dim, args.projection_dim, action_shape, args.hidden_dim,  
                                        state_shape,args.hidden_dim_state).cuda()
        self.critic_target = m.EfficientCritic(self.encoder_rl.out_dim, args.projection_dim, action_shape,
                                               args.hidden_dim,  state_shape, args.hidden_dim_state).cuda()
        
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.log_alpha = torch.tensor(np.log(args.init_temperature)).cuda()
        self.log_alpha.requires_grad = True
        self.target_entropy = -np.prod(action_shape)

        self.actor_optimizer = torch.optim.Adam(
            itertools.chain(self.actor.parameters(), self.encoder_rl.parameters()), lr=args.actor_lr, betas=(args.actor_beta, 0.999)
        )
        self.critic_optimizer = torch.optim.Adam(
            itertools.chain(self.critic.parameters(), self.encoder_rl.parameters()), lr=args.critic_lr, betas=(args.critic_beta, 0.999)
        )
        self.log_alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=args.alpha_lr, betas=(args.alpha_beta, 0.999)
        )
        
        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    def eval(self):
        self.train(False)
    def save_camera_intrinsic(self, intrinsic):
        return
    @property
    def alpha(self):
        return self.log_alpha.exp()
        
    def _obs_to_input(self, obs):
        if isinstance(obs, utils.LazyFrames) or len(obs.shape) == 3:
            obs = torch.FloatTensor(obs).cuda().div(255)
            return obs.unsqueeze(0)

        else:
            obs = torch.FloatTensor(obs).cuda().unsqueeze(0)
            return obs

    def select_action(self, obs, state=None):
        _obs = self._obs_to_input(obs[:3])
        if state is not None:
            state = self._obs_to_input(state)
        with torch.no_grad():
            mu, _, _, _ = self.actor(self.encoder_rl(_obs), state, compute_pi=False, compute_log_pi=False)
        return mu.cpu().data.numpy().flatten()

    def sample_action(self, obs, state=None, step=None):
        _obs = self._obs_to_input(obs[:3])
        if state is not None:
            state = self._obs_to_input(state)
        with torch.no_grad():
            mu, pi, _, _ = self.actor(self.encoder_rl(_obs), state, compute_log_pi=False)
        return pi.cpu().data.numpy().flatten()

    def update_critic(self, obs, state, action, reward, next_obs, next_state, L=None, step=None):
        with torch.no_grad():
            _, policy_action, log_pi, _ = self.actor(self.encoder_rl(next_obs), next_state)
            target_Q1, target_Q2 = self.critic_target(self.encoder_rl(next_obs), next_state, policy_action)
            target_V = torch.min(target_Q1,
                                    target_Q2) - self.alpha.detach() * log_pi
            target_Q = reward + (self.discount * target_V)

        current_Q1, current_Q2 = self.critic(self.encoder_rl(obs), state, action)
        critic_loss = F.mse_loss(current_Q1,
                                    target_Q) + F.mse_loss(current_Q2, target_Q)
        if L is not None:
            L.log('train_critic/loss', critic_loss, step)
        if self.args.use_wandb and not self.args.remove_addition_log:
            wandb.log({'train/critic_loss':critic_loss},step=step+1)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def update_actor_and_alpha(self, obs, state, L=None, step=None, update_alpha=True):
        _, pi, log_pi, log_std = self.actor(self.encoder_rl(obs), state)
        actor_Q1, actor_Q2 = self.critic(self.encoder_rl(obs), state, pi)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_pi - actor_Q).mean()

        if L is not None:
            L.log('train_actor/loss', actor_loss, step)
            entropy = 0.5 * log_std.shape[1] * (1.0 + np.log(2 * np.pi)
                                                ) + log_std.sum(dim=-1)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        if self.args.use_wandb and not self.args.remove_addition_log:
            wandb.log({'train/actor_loss':actor_loss},step=step+1)

        if update_alpha:
            self.log_alpha_optimizer.zero_grad()
            alpha_loss = (self.alpha * (-log_pi - self.target_entropy).detach()).mean()

            if L is not None:
                L.log('train_alpha/loss', alpha_loss, step)
                L.log('train_alpha/value', self.alpha, step)

            alpha_loss.backward()
            self.log_alpha_optimizer.step()


    def update(self, replay_buffer, L, step):
        if not self.use_prioritized_buffer:
            obs, state, action, reward, next_obs, next_state, camera_param = replay_buffer.sample()
        else:
            obs, state, action, reward, next_obs, next_state, camera_param, indices, weights = replay_buffer.sample()

        obs = obs[:,:3]
        next_obs = next_obs[:,:3]

        self.update_critic(obs, state, action, reward, next_obs,next_state, L, step)

        self.update_actor_and_alpha(obs, state, L, step)
        utils.soft_update_params(self.critic, self.critic_target, self.tau)
