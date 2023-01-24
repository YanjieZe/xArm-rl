import math
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



def _get_out_shape(in_shape, layers):
    x = torch.randn(*in_shape).unsqueeze(0)
    return layers(x).squeeze(0).shape


def gaussian_logprob(noise, log_std):
    """Compute Gaussian log probability"""
    residual = (-0.5 * noise.pow(2) - log_std).sum(-1, keepdim=True)
    return residual - 0.5 * np.log(2 * np.pi) * noise.size(-1)


def squash(mu, pi, log_pi):
    """Apply squashing function, see appendix C from https://arxiv.org/pdf/1812.05905.pdf"""
    mu = torch.tanh(mu)
    if pi is not None:
        pi = torch.tanh(pi)
    if log_pi is not None:
        log_pi -= torch.log(F.relu(1 - pi.pow(2)) + 1e-6).sum(-1, keepdim=True)
    return mu, pi, log_pi



def orthogonal_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)


def xavier_uniform_init(module, gain=1.0):
    if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
        nn.init.xavier_uniform_(module.weight.data, gain)
        nn.init.constant_(module.bias.data, 0)
    return module


class Flatten(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class Identity(nn.Module):
    def __init__(self, obs_shape=None, out_dim=None):
        super().__init__()
        self.out_shape = obs_shape
        self.out_dim = out_dim

    def forward(self, x):
        return x


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


class Encoder(nn.Module):
    def __init__(self, shared_cnn, head_cnn, projection):
        super().__init__()
        self.shared_cnn = shared_cnn
        self.head_cnn = head_cnn
        self.projection = projection
        self.out_dim = projection.out_dim

    def forward(self, x, detach=False):
        x = self.shared_cnn(x)
        x = self.head_cnn(x)

        if detach:
            x = x.detach()
        return self.projection(x)


class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        #self.conv_query = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)
        self.fc_query = nn.Linear(in_features=in_channels, out_features=in_channels)
        self.conv_key = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)
        self.conv_value = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)
        self.in_channels = in_channels

    def forward(self, query, key, value):
        N, C, H, W = value.shape
        #assert query.shape == key.shape == value.shape, "Key, query and value inputs must be of the same dimensions in this implementation"
        q = self.fc_query(query).reshape(N, C, 1)  # .permute(0, 2, 1)
        k = self.conv_key(key).reshape(N, C, H * W)  # .permute(0, 2, 1)
        v = self.conv_value(value).reshape(N, C, H * W)  # .permute(0, 2, 1)
        attention = k.transpose(1, 2) @ q / C ** 0.5
        attention = attention.softmax(dim=1)
        output = v @ attention
        output = output.reshape(N, C, H, W)
        return query + output  # Add with query and output


class AttentionBlock(nn.Module):
    def __init__(self, dim, num_heads=1, qkv_bias=False, qk_scale=None, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)
        self.attn = SelfAttention(dim[0])
        self.context = contextualReasoning
        temp_shape = _get_out_shape(dim, self.attn, attn=True)
        self.out_shape = _get_out_shape(temp_shape, nn.Flatten())
        self.apply(orthogonal_init)

    def forward(self, query, key, value):
        x = self.attn(self.norm1(query), self.norm2(key), self.norm3(value))
        return x

class NormalizeImg(nn.Module):
    def __init__(self, mean_zero=False):
        super().__init__()
        self.mean_zero = mean_zero

    def forward(self, x):
        if self.mean_zero:
            return x/255. - 0.5
        return x/255.

class SharedCNN(nn.Module):
    def __init__(self, obs_shape, num_layers=11, num_filters=32, mean_zero=False):
        super().__init__()
        assert len(obs_shape) == 3
        self.num_layers = num_layers
        self.num_filters = num_filters
        self.layers = [nn.Conv2d(obs_shape[0], num_filters, 3, stride=2)]
        for _ in range(1, num_layers):
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Conv2d(num_filters, num_filters, 3, stride=1))
        self.layers = nn.Sequential(*self.layers)
        self.out_shape = _get_out_shape(obs_shape, self.layers)
        self.apply(orthogonal_init)

    def forward(self, x):
        return self.layers(x)

class SharedCNN2(nn.Module):
    def __init__(self, obs_shape, in_shape, num_layers=11, num_filters=32, project=False, project_conv=False, attention=False):
        super().__init__()

        self.num_layers = num_layers
        self.num_filters = num_filters
        self.project = project
        self.project_conv = project_conv

        self.attention = attention


        if self.attention:
            self.layers_obs = [nn.Conv2d(in_shape[0], num_filters, 3, stride=2)]
            for _ in range(1, num_layers):
                self.layers_obs.append(nn.ReLU())
                self.layers_obs.append(nn.Conv2d(num_filters, num_filters, 3, stride=1))
            self.layers_obs.append(nn.ReLU())
            self.layers_obs = nn.Sequential(*self.layers_obs)
            self.out_shape = _get_out_shape(obs_shape, self.layers_obs)

        self.layers = [Identity()]
        if project and self.project_conv:
            self.layers = [nn.Conv2d(obs_shape[0], 32, kernel_size=1, stride=1, padding=0), nn.ReLU()]
            if self.attention:
                self.layers.append(nn.Conv2d(32, self.out_shape[0], kernel_size=3, stride=1, padding=1))
                self.layers.append(nn.ReLU())
            self.layers = nn.Sequential(*self.layers)

        self.layers = nn.Sequential(*self.layers)

        self.out_shape = _get_out_shape(obs_shape, self.layers)

        #self.attn =

        self.apply(orthogonal_init)

    def forward(self, x):
        return self.layers(x)



class HeadCNN(nn.Module):
    def __init__(self, in_shape, num_layers=0, num_filters=32):
        super().__init__()
        self.layers = []
        for _ in range(0, num_layers):
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Conv2d(num_filters, num_filters, 3, stride=1))
        self.layers.append(Flatten())
        self.layers = nn.Sequential(*self.layers)

        self.out_shape = _get_out_shape(in_shape, self.layers)
        self.apply(orthogonal_init)

    def forward(self, x):
        return self.layers(x)

class EfficientActorMVP(nn.Module):
    def __init__(self, out_dim, projection_dim, action_shape, hidden_dim, log_std_min, log_std_max, state_shape=None, hidden_dim_state=None):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max


        self.trunk = nn.Sequential(nn.Linear(out_dim, 256), nn.SELU())

        self.layers = nn.Sequential( 
            nn.Linear(256, 128), nn.SELU(inplace=True),
            nn.Linear(128, 64), nn.SELU(inplace=True),
            nn.Linear(64, 2 * action_shape[0])
        )


        self.state_encoder = None

        self.apply(orthogonal_init)

    def forward(self, x, state, compute_pi=True, compute_log_pi=True):
        try:
            x = self.trunk(x)
        except:
            import pdb; pdb.set_trace()

        mu, log_std = self.layers(x).chunk(2, dim=-1)
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (
                self.log_std_max - self.log_std_min
        ) * (log_std + 1)

        if compute_pi:
            std = log_std.exp()
            noise = torch.randn_like(mu)
            pi = mu + noise * std
        else:
            pi = None
            entropy = None

        if compute_log_pi:
            log_pi = gaussian_logprob(noise, log_std)
        else:
            log_pi = None

        mu, pi, log_pi = squash(mu, pi, log_pi)

        return mu, pi, log_pi, log_std


        
        
class EfficientActor(nn.Module):
    def __init__(self, out_dim, projection_dim, action_shape, hidden_dim, log_std_min, log_std_max, state_shape=None, hidden_dim_state=None):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max



        self.trunk = nn.Sequential(nn.Linear(out_dim, projection_dim),
								   nn.LayerNorm(projection_dim), nn.Tanh())

        self.layers = nn.Sequential( 
            nn.Linear(projection_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2 * action_shape[0])
        )

        if state_shape:
            self.state_encoder = nn.Sequential(nn.Linear(state_shape[0], hidden_dim_state),
                                                nn.ReLU(inplace=True),
                                                nn.Linear(hidden_dim_state, projection_dim),
                                                nn.LayerNorm(projection_dim), nn.Tanh())
        else:
            self.state_encoder = None

        self.apply(orthogonal_init)

    def forward(self, x, state, compute_pi=True, compute_log_pi=True):
        try:
            x = self.trunk(x)
        except:
            import pdb; pdb.set_trace()
        if self.state_encoder:
            x = x + self.state_encoder(state)

        mu, log_std = self.layers(x).chunk(2, dim=-1)
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (
                self.log_std_max - self.log_std_min
        ) * (log_std + 1)

        if compute_pi:
            std = log_std.exp()
            noise = torch.randn_like(mu)
            pi = mu + noise * std
        else:
            pi = None
            entropy = None

        if compute_log_pi:
            log_pi = gaussian_logprob(noise, log_std)
        else:
            log_pi = None

        mu, pi, log_pi = squash(mu, pi, log_pi)

        return mu, pi, log_pi, log_std


class QFunction(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.apply(orthogonal_init)

    def forward(self, obs, action):
        assert obs.size(0) == action.size(0)
        return self.trunk(torch.cat([obs, action], dim=1))

class EfficientCriticMVP(nn.Module):
    def __init__(self, out_dim, projection_dim, action_shape, hidden_dim,  state_shape=None, hidden_dim_state=None):
        super().__init__()
        self.projection = nn.Sequential(nn.Linear(out_dim, 256), nn.SELU(),
                   nn.Linear(256, 128),nn.SELU() )
        self.state_encoder = None

        self.Q1 = nn.Sequential(
            nn.Linear(128 + action_shape[0], 64), nn.SELU(),
            nn.Linear(64, 1))
        self.Q2 = nn.Sequential(
            nn.Linear(128 + action_shape[0], 64), nn.SELU(),
            nn.Linear(64, 1))

        self.apply(orthogonal_init)

    def forward(self, obs, state, action):
        obs = self.projection(obs)

        if self.state_encoder:
            obs = obs + self.state_encoder(state)

        h = torch.cat([obs, action], dim=-1)
        return self.Q1(h), self.Q2(h)


class EfficientCritic(nn.Module):
    def __init__(self, out_dim, projection_dim, action_shape, hidden_dim,  state_shape=None, hidden_dim_state=None):
        super().__init__()
        self.projection = nn.Sequential(nn.Linear(out_dim, projection_dim),
                                        nn.LayerNorm(projection_dim), nn.Tanh())
        if state_shape:
            self.state_encoder = nn.Sequential(nn.Linear(state_shape[0], hidden_dim_state),
                                                nn.ReLU(inplace=True),
                                                nn.Linear(hidden_dim_state, projection_dim),
                                                nn.LayerNorm(projection_dim), nn.Tanh())
        else:
            self.state_encoder = None

        self.Q1 = nn.Sequential(
            nn.Linear(projection_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))
        self.Q2 = nn.Sequential(
            nn.Linear(projection_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))
        self.apply(orthogonal_init)

    def forward(self, obs, state, action):
        obs = self.projection(obs)

        if self.state_encoder:
            obs = obs + self.state_encoder(state)

        h = torch.cat([obs, action], dim=-1)
        return self.Q1(h), self.Q2(h)


class UnparametricNorm(nn.Module):
	def __init__(self, dim=1, eps=1e-10, scale=10.):
		super().__init__()
		self.dim = dim
		self.eps = eps
		self.scale = scale
	
	def forward(self, x):
		return F.normalize(x, dim=self.dim, eps=self.eps) * self.scale
                
class MLP(nn.Module):
	def __init__(self, in_dim, hidden_dims, out_dim, activation_fn=nn.ELU, penultimate_norm=False):
		super().__init__()
		assert isinstance(hidden_dims, (tuple, list, int)), \
			f'expected hidden dims to be int, tuple, or list, received {hidden_dims}'
		if isinstance(hidden_dims, int):
			hidden_dims = [hidden_dims]
		self.in_dim = in_dim
		self.hidden_dims = hidden_dims
		self.out_dim = out_dim
		layer_dims = [in_dim] + list(hidden_dims) + [out_dim]
		layers = []
		for i in range(1, len(layer_dims)-1):
			layers.extend([nn.Linear(layer_dims[i - 1], layer_dims[i]), activation_fn()])
		if penultimate_norm:
			layers.append(UnparametricNorm())
		layers.append(nn.Linear(layer_dims[-2], layer_dims[-1]))
		self.layers = nn.Sequential(*layers)
		self.apply(orthogonal_init)
	
	def forward(self, x):
		return self.layers(x)

class RLProjection(nn.Module):
	def __init__(self, in_shape, out_dim):
		super().__init__()
		self.out_dim = out_dim
		self.projection = nn.Sequential(
			nn.Linear(in_shape[0], out_dim),
			nn.LayerNorm(out_dim),
			nn.Tanh()
		)
		self.apply(orthogonal_init)
	
	def forward(self, x):
		return self.projection(x)

class Actor(nn.Module):
	def __init__(self, encoder, action_shape, hidden_dim, log_std_min, log_std_max, multiview=False):
		super().__init__()
		self.encoder = encoder
		self.log_std_min = log_std_min
		self.log_std_max = log_std_max
		self.mlp = nn.Sequential(
			nn.Linear(self.encoder.out_dim, hidden_dim), nn.ReLU(inplace=True),
			nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True),
			nn.Linear(hidden_dim, 2 * action_shape[0])
		)
		self.mlp.apply(orthogonal_init)
		self.multiview = multiview

	def forward(self, x_in, compute_pi=True, compute_log_pi=True, detach=False):
		if self.multiview:
			x1, x2 = x_in[:,:3,:,:], x_in[:,3:6,:,:]
			x = self.encoder(x1, x2, detach)
		else:
			x = self.encoder(x_in, detach)
			
		mu, log_std = self.mlp(x).chunk(2, dim=-1)
		log_std = torch.tanh(log_std)
		log_std = self.log_std_min + 0.5 * (
			self.log_std_max - self.log_std_min
		) * (log_std + 1)

		if compute_pi:
			std = log_std.exp()
			noise = torch.randn_like(mu)
			pi = mu + noise * std
		else:
			pi = None
			entropy = None

		if compute_log_pi:
			log_pi = gaussian_logprob(noise, log_std)
		else:
			log_pi = None

		mu, pi, log_pi = squash(mu, pi, log_pi)

		return mu, pi, log_pi, log_std

class Critic(nn.Module):
	def __init__(self, encoder, action_shape, hidden_dim, multiview=False):
		super().__init__()
		self.encoder = encoder
		self.Q1 = QFunction(
			self.encoder.out_dim, action_shape[0], hidden_dim
		)
		self.Q2 = QFunction(
			self.encoder.out_dim, action_shape[0], hidden_dim
		)
		self.multiview = multiview

	def forward(self, x_in, action, detach=False):
		if self.multiview:
			x1, x2 = x_in[:,:3,:,:], x_in[:,3:6,:,:]
			x = self.encoder(x1, x2, detach)
		else:
			x = self.encoder(x_in, detach)

		return self.Q1(x, action), self.Q2(x, action)
