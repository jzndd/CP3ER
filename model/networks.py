import numpy as np
import torch
import torch.nn as nn
from .cm import ConsistencyModel

import torch.nn.functional as F

import utils

# Critic Network
class MoGCritic(nn.Module):
    """
    A network module designed to function as a critic in a reinforcement learning framework,
    adhering to a representation of a mixture of Gaussian (MoG) distributions.This network is designed to predict parameters for a mixture of Gaussian distributions
    based on the given observation and action inputs.

    Parameters:
    - repr_dim (int): Dimensionality of the input observation representations.
    - action_dim (int): Dimensionality of the action space.
    - feature_dim (int): Intermediate feature dimensionality of the network.
    - hidden_dim (int): Dimensionality of the hidden layers.
    - num_groups (int, optional): Number of groups for GroupNorm. If set to 0, LayerNorm is used instead.
    - num_components (int, optional): Number of components in the Gaussian mixture.
    - init_scale (float, optional): Initial scaling factor for the standard deviations of the Gaussians.

    Output:
    - forward(obs, action): Processes the input observation and action through the network and returns the
      means, standard deviations, and logits for each Gaussian component in the mixture.
    """
    def __init__(self, repr_dim,action_dim, feature_dim, hidden_dim, num_groups=0, num_components=1, init_scale=1e-3):
        super(MoGCritic, self).__init__()
        # self.structure = structure
        self.num_components = num_components
        self.init_scale = init_scale
        self.num_groups = num_groups

        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                    nn.LayerNorm(feature_dim))
        
        self.linear1 = nn.Linear(feature_dim + action_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        if self.num_groups:
            self.gn1 = nn.GroupNorm(num_groups,hidden_dim)
            self.gn2 = nn.GroupNorm(num_groups,hidden_dim)
        else:
            self.ln1 = nn.LayerNorm(hidden_dim)
            self.ln2 = nn.LayerNorm(hidden_dim)

        self.mus = nn.Linear(hidden_dim, num_components)
        self.stdevs = nn.Linear(hidden_dim, num_components)
        self.logits = nn.Linear(hidden_dim, num_components)

    def forward(self, obs, action):
        info = {}
        h = self.trunk(obs)
        x = torch.cat([h, action], dim=-1)
        if self.num_groups:
            x = torch.relu(self.gn1(self.linear1(x)))
            x = torch.relu(self.gn2(self.linear2(x)))
        else:
            x = torch.relu(self.ln1(self.linear1(x)))
            x = torch.relu(self.ln2(self.linear2(x)))

        mus = self.mus(x).unsqueeze(1)
        stdevs = self.init_scale * F.softplus(self.stdevs(x)) / F.softplus(torch.tensor(0.)) + 1e-4
        stdevs = stdevs.unsqueeze(1)
        logits = self.logits(x).unsqueeze(1)

        info['mus'] = mus
        info['stdevs'] = stdevs
        info['logits'] = logits

        return info
 
# Actor Network
class CPActor(nn.Module):
    """
    A network module designed to function as an actor in a reinforcement learning framework,
    adhering to a specified consistency policy (CP). This actor network outputs action values
    given state inputs by utilizing an internal consistency model.

    Parameters:
    - repr_dim (int): Dimensionality of the state representation input to the model.
    - action_dim (int): Dimensionality of the action space.
    - device (str, optional): The device (e.g., 'cuda:1') on which the model computations will be performed.

    Output:
    - forward(state, return_dict=False): Processes the input state through the ConsistencyModel.
    """

    def __init__(self, repr_dim, action_dim, device, feature_dim, hidden_dim):

        super(CPActor, self).__init__()

        self.device = device

        self.cm = ConsistencyModel(state_dim=repr_dim, action_dim=action_dim, device=device,
                                   feature_dim=feature_dim, hidden_dim=hidden_dim)
        self.to(device)

    def forward(self, state):
        return self.cm(state)
    
    def to(self, device):
        super(CPActor, self).to(device)
    
    def loss(self, action, state):
        return self.cm.loss(action, state)

# Encoder network
class Encoder(nn.Module):
    def __init__(self, obs_shape):
        super().__init__()

        assert len(obs_shape) == 3
        self.repr_dim = 32 * 35 * 35

        self.convnet = nn.Sequential(nn.Conv2d(obs_shape[0], 32, 3, stride=2),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU())

        self.apply(utils.weight_init)

    def forward(self, obs):
        obs = obs / 255.0 - 0.5
        h = self.convnet(obs)
        h = h.view(h.shape[0], -1)
        return h
    

# ablation Study of Critic Chosen
class Critic(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())

        self.Q1 = nn.Sequential(
            nn.Linear(feature_dim + action_shape, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

        self.Q2 = nn.Sequential(
            nn.Linear(feature_dim + action_shape, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

        self.apply(utils.weight_init)

    def forward(self, obs, action):
        h = self.trunk(obs)
        h_action = torch.cat([h, action], dim=-1)
        q1 = self.Q1(h_action)
        q2 = self.Q2(h_action)

        return q1, q2