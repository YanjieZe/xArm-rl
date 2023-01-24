import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
import utils
import algorithms.modules as m
import augmentations


class SACv2(object):
    def __init__(self, obs_shape, state_shape, action_shape, args):
        self.discount = args.discount
        self.update_freq = args.update_freq
        self.tau = args.tau

        self.augmentation = args.augmentation

        self.state_obs = True if args.observation_type == "state+image" else False
        self.observation_type = args.observation_type

        if (self.state_obs==False):
            state_shape = None

        if args.observation_type == "state":
            pass
        elif args.observation_type in ["image", "state+image"]:
            obs_shape = (3, obs_shape[1], obs_shape[2])
        
        if args.observation_type == "state":
            shared = m.Identity(obs_shape)
            head = m.Identity(obs_shape)
        elif args.observation_type in ["image", "state+image"]:
            shared = m.SharedCNN(obs_shape, args.num_shared_layers, args.num_filters, args.mean_zero)
            head = m.HeadCNN(shared.out_shape, args.num_head_layers, args.num_filters)
        
        self.encoder = m.Encoder(
            shared,
            head,
            m.Identity(out_dim=head.out_shape[0])
        ).cuda()

        
        self.state_predictor = None
        self.predictor_optimizer = None

        self.actor = m.EfficientActor(self.encoder.out_dim, args.projection_dim, action_shape, args.hidden_dim, args.actor_log_std_min, args.actor_log_std_max, state_shape, args.hidden_dim_state).cuda()
        self.critic = m.EfficientCritic(self.encoder.out_dim, args.projection_dim, action_shape, args.hidden_dim, state_shape, args.hidden_dim_state).cuda()
        self.critic_target = m.EfficientCritic(self.encoder.out_dim, args.projection_dim, action_shape, args.hidden_dim, state_shape, args.hidden_dim_state).cuda()
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.log_alpha = torch.tensor(np.log(args.init_temperature)).cuda()
        self.log_alpha.requires_grad = True
        self.target_entropy = -np.prod(action_shape)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.lr)
        self.critic_optimizer = torch.optim.Adam(itertools.chain(self.encoder.parameters(), self.critic.parameters()), lr=args.lr)
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=args.alpha_lr, betas=(args.alpha_beta, 0.999))


        self.aug = m.RandomShiftsAug(pad=4)
        self.train()
        print('Encoder:', utils.count_parameters(self.encoder))
        print('Actor:', utils.count_parameters(self.actor))
        print('Critic:', utils.count_parameters(self.critic))

    def train(self, training=True):
        self.training = training
        for p in [self.encoder, self.actor, self.critic, self.critic_target]:#, self.state_predictor]:
            p.train(training)
            # if p is not None:
            # 	p.train(training)
    def save_camera_intrinsic(self, intrinsic):
        return
    def eval(self):
        self.train(False)

    @property
    def alpha(self):
        return self.log_alpha.exp()
        
    def _obs_to_input(self, obs):
        if isinstance(obs, utils.LazyFrames):
            _obs = np.array(obs)
        else:
            _obs = obs
        _obs = torch.FloatTensor(_obs).cuda()
        _obs = _obs.unsqueeze(0)
        return _obs

    def select_action(self, obs, state=None):
        if self.observation_type == "state":
            _obs = self._obs_to_input(obs)
        elif self.observation_type in ["image", "state+image"]:
            obs = obs[:3]

            _obs = self._obs_to_input(obs).div(255)
        if state is not None:
            state = self._obs_to_input(state)
        with torch.no_grad():
            obs = self.encoder(_obs)
            mu, _, _, _ = self.actor(obs, state, compute_pi=False, compute_log_pi=False)
        return mu.cpu().data.numpy().flatten()

    def sample_action(self, obs, state=None, step=None):
        if self.observation_type == "state":
            _obs = self._obs_to_input(obs)
        elif self.observation_type in ["image", "state+image"]:
            obs = obs[:3]
            _obs = self._obs_to_input(obs).div(255)
        if state is not None:
            state = self._obs_to_input(state)
        with torch.no_grad():
            obs = self.encoder(_obs)
            mu, pi, _, _ = self.actor(obs, state, compute_log_pi=False)
        return pi.cpu().data.numpy().flatten()

    def update_critic(self, obs, state, action, reward, next_obs, next_state, L=None, step=None):
        
        with torch.no_grad():
            _, policy_action, log_pi, _ = self.actor(next_obs, next_state)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_state, policy_action)
            target_V = torch.min(target_Q1,
                                    target_Q2) - self.alpha.detach() * log_pi
            target_Q = reward + (self.discount * target_V)

        Q1, Q2 = self.critic(obs, state, action)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)
        if L is not None:
            L.log('train_critic/loss', critic_loss, step)

        self.critic_optimizer.zero_grad(set_to_none=True)

        if self.state_predictor:
            self.state_predictor.zero_grad(set_to_none=True)

            predicted = self.state_predictor(obs)
            predictor_loss = F.mse_loss(predicted, state)
            if L is not None:
                L.log('train_predictor/loss', predictor_loss, step)
            (predictor_loss+critic_loss).backward()
            self.predictor_optimizer.step()
        else:
            critic_loss.backward()
        self.critic_optimizer.step()

    def update_actor_and_alpha(self, obs, state, L=None, step=None, update_alpha=True):
        _, pi, log_pi, log_std = self.actor(obs, state)
        Q1, Q2 = self.critic(obs, state, pi)
        Q = torch.min(Q1, Q2)
        actor_loss = (self.alpha.detach() * log_pi - Q).mean()
        if L is not None:
            L.log('train_actor/loss', actor_loss, step)

        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()

        if update_alpha:
            self.log_alpha_optimizer.zero_grad(set_to_none=True)
            alpha_loss = (self.alpha * (-log_pi - self.target_entropy).detach()).mean()

            if L is not None:
                L.log('train_alpha/loss', alpha_loss, step)
                L.log('train_alpha/value', self.alpha, step)

            alpha_loss.backward()
            self.log_alpha_optimizer.step()

    def update(self, replay_buffer, L, step):
        if step % self.update_freq != 0:
            return

        obs, state, action, reward, next_obs, next_state, camera_param = replay_buffer.sample()
        if self.observation_type == "state":
            pass
        elif self.observation_type in ["image", "state+image"]:
            obs = self.aug(obs)
            next_obs = self.aug(next_obs)
                
            if self.augmentation=='colorjitter':
                obs = augmentations.random_color_jitter(obs)
                next_obs = augmentations.random_color_jitter(next_obs) 
            elif self.augmentation=='noise':
                obs = augmentations.random_noise(obs)
                next_obs = augmentations.random_noise(next_obs)

            obs = obs[:, :3]
            next_obs = next_obs[:, :3]


        obs = self.encoder(obs)

        with torch.no_grad():
            next_obs = self.encoder(next_obs)


        self.update_critic(obs, state, action, reward, next_obs, next_state, L, step)
        self.update_actor_and_alpha(obs.detach(), state, L, step)
        utils.soft_update_params(self.critic, self.critic_target, self.tau)