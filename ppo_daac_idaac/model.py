import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from ppo_daac_idaac.utils import init
from ppo_daac_idaac.distributions import Categorical
from ppo_daac_idaac.distributions import FixedCategorical


init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                        constant_(x, 0))

init_relu_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                        constant_(x, 0), nn.init.calculate_gain('relu'))


def apply_init_(modules):
    """
    Initialize NN modules
    """
    for m in modules:
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class Flatten(nn.Module):
    """
    Flatten a tensor
    """
    def forward(self, x):
        return x.reshape(x.size(0), -1)


class Conv2d_tf(nn.Conv2d):
    """
    Conv2d with the padding behavior from TF
    """
    def __init__(self, *args, **kwargs):
        super(Conv2d_tf, self).__init__(*args, **kwargs)
        self.padding = kwargs.get("padding", "SAME")

    def _compute_padding(self, input, dim):
        input_size = input.size(dim + 2)
        filter_size = self.weight.size(dim + 2)
        effective_filter_size = (filter_size - 1) * self.dilation[dim] + 1
        out_size = (input_size + self.stride[dim] - 1) // self.stride[dim]
        total_padding = max(
            0, (out_size - 1) * self.stride[dim] + effective_filter_size - input_size
        )
        additional_padding = int(total_padding % 2 != 0)

        return additional_padding, total_padding

    def forward(self, input):
        if self.padding == "VALID":
            return F.conv2d(
                input,
                self.weight,
                self.bias,
                self.stride,
                padding=0,
                dilation=self.dilation,
                groups=self.groups,
            )
        rows_odd, padding_rows = self._compute_padding(input, dim=0)
        cols_odd, padding_cols = self._compute_padding(input, dim=1)
        if rows_odd or cols_odd:
            input = F.pad(input, [0, cols_odd, 0, rows_odd])

        return F.conv2d(
            input,
            self.weight,
            self.bias,
            self.stride,
            padding=(padding_rows // 2, padding_cols // 2),
            dilation=self.dilation,
            groups=self.groups,
        )

class NNBase(nn.Module):
    """
    Actor-Critic network (base class)
    """
    def __init__(self, hidden_size):
        super(NNBase, self).__init__()

        self._hidden_size = hidden_size

    @property
    def output_size(self):
        return self._hidden_size


class BasicBlock(nn.Module):
    """
    Residual Network Block
    """
    def __init__(self, n_channels, stride=1):
        super(BasicBlock, self).__init__()

        self.conv1 = Conv2d_tf(n_channels, n_channels, kernel_size=3, stride=1, padding=(1,1))
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = Conv2d_tf(n_channels, n_channels, kernel_size=3, stride=1, padding=(1,1))
        self.stride = stride

        apply_init_(self.modules())

        self.train()

    def forward(self, x):
        identity = x

        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)

        out += identity
        return out


class ResNetBase(NNBase):
    """
    Residual Network 
    """
    def __init__(self, num_inputs, hidden_size=256, channels=[16,32,32]):
        super(ResNetBase, self).__init__(hidden_size)

        self.layer1 = self._make_layer(num_inputs, channels[0])
        self.layer2 = self._make_layer(channels[0], channels[1])
        self.layer3 = self._make_layer(channels[1], channels[2])

        self.flatten = Flatten()
        self.relu = nn.ReLU()

        self.fc = init_relu_(nn.Linear(2048, hidden_size))
        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        apply_init_(self.modules())

        self.train()

    def _make_layer(self, in_channels, out_channels, stride=1):
        layers = []

        layers.append(Conv2d_tf(in_channels, out_channels, kernel_size=3, stride=1))
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        layers.append(BasicBlock(out_channels))
        layers.append(BasicBlock(out_channels))

        return nn.Sequential(*layers)

    def forward(self, inputs):
        x = inputs

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.relu(self.flatten(x))
        x = self.relu(self.fc(x))

        return self.critic_linear(x), x

class PolicyResNetBase(NNBase):
    """
    Residual Network 
    """
    def __init__(self, num_inputs, hidden_size=256, channels=[16,32,32], num_actions=15):
        super(PolicyResNetBase, self).__init__(hidden_size)
        self.num_actions = num_actions 

        self.layer1 = self._make_layer(num_inputs, channels[0])
        self.layer2 = self._make_layer(channels[0], channels[1])
        self.layer3 = self._make_layer(channels[1], channels[2])

        self.flatten = Flatten()
        self.relu = nn.ReLU()

        self.fc = init_relu_(nn.Linear(2048, hidden_size))
        self.critic_linear = init_(nn.Linear(hidden_size + num_actions, 1))

        apply_init_(self.modules())

        self.train()

    def _make_layer(self, in_channels, out_channels, stride=1):
        layers = []

        layers.append(Conv2d_tf(in_channels, out_channels, kernel_size=3, stride=1))
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        layers.append(BasicBlock(out_channels))
        layers.append(BasicBlock(out_channels))

        return nn.Sequential(*layers)

    def forward(self, inputs, actions=None):
        x = inputs

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.relu(self.flatten(x))
        x = self.relu(self.fc(x))

        if actions is None:
           onehot_actions = torch.zeros(x.shape[0], self.num_actions).to(x.device)
        else:
            onehot_actions = F.one_hot(actions.squeeze(1), self.num_actions).float()
        gae_inputs = torch.cat((x, onehot_actions), dim=1)

        return self.critic_linear(gae_inputs), x

class ValueResNet(NNBase):
    """
    Residual Network 
    """
    def __init__(self, num_inputs, hidden_size=256, channels=[16,32,32]):
        super(ValueResNet, self).__init__(hidden_size)

        self.layer1 = self._make_layer(num_inputs, channels[0])
        self.layer2 = self._make_layer(channels[0], channels[1])
        self.layer3 = self._make_layer(channels[1], channels[2])

        self.flatten = Flatten()
        self.relu = nn.ReLU()

        self.fc = init_relu_(nn.Linear(2048, hidden_size))
        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        apply_init_(self.modules())

        self.train()

    def _make_layer(self, in_channels, out_channels, stride=1):
        layers = []

        layers.append(Conv2d_tf(in_channels, out_channels, kernel_size=3, stride=1))
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        layers.append(BasicBlock(out_channels))
        layers.append(BasicBlock(out_channels))

        return nn.Sequential(*layers)

    def forward(self, inputs):
        x = inputs

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.relu(self.flatten(x))
        x = self.relu(self.fc(x))

        return self.critic_linear(x)


class LinearOrderClassifier(nn.Module):
    def __init__(self, emb_size=256):
        super(LinearOrderClassifier, self).__init__()
        self.main = nn.Sequential(
            Flatten(),
            init_(nn.Linear(2*emb_size, 2)), 
            nn.Softmax(dim=1),
        )
        self.train()

    def forward(self, emb):
        x = self.main(emb)
        return x


class NonlinearOrderClassifier(nn.Module):
    def __init__(self, emb_size=256, hidden_size=4):
        super(NonlinearOrderClassifier, self).__init__()
        self.main = nn.Sequential(
            Flatten(),
            init_relu_(nn.Linear(2*emb_size, hidden_size)), nn.ReLU(),
            init_(nn.Linear(hidden_size, 2)), 
            nn.Softmax(dim=1),
        )
        self.train()

    def forward(self, emb):
        x = self.main(emb)
        return x


class PPOnet(nn.Module):
    """
    PPO netowrk 
    """
    def __init__(self, obs_shape, num_actions, base_kwargs=None):
        super(PPOnet, self).__init__()
        
        if base_kwargs is None:
            base_kwargs = {}
        
        base = ResNetBase
        
        self.base = base(obs_shape[0], **base_kwargs)
        self.dist = Categorical(self.base.output_size, num_actions)
        
    def forward(self, inputs):
        raise NotImplementedError

    def act(self, inputs, deterministic=False):
        value, actor_features = self.base(inputs)
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action, action_log_probs

    def get_value(self, inputs):
        value, _ = self.base(inputs)
        return value

    def evaluate_actions(self, inputs, action):
        value, actor_features = self.base(inputs)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()
        
        return value, action_log_probs, dist_entropy


class IDAACnet(nn.Module):
    """
    IDAAC network
    """
    def __init__(self, obs_shape, num_actions, base_kwargs=None):
        super(IDAACnet, self).__init__()
        
        if base_kwargs is None:
            base_kwargs = {}
        
        base = PolicyResNetBase
        
        self.base = base(obs_shape[0], **base_kwargs)
        self.value_net = ValueResNet(obs_shape[0], **base_kwargs)
        self.dist = Categorical(self.base.output_size, num_actions)
        
    def forward(self, inputs):
        raise NotImplementedError

    def run_actor(self, x):
        gae, actor_features = self.base(x)
        dist = self.dist(actor_features)

        return dist, gae, actor_features

    def act(self, inputs, deterministic=False):
        dist, _, _ = self.run_actor(inputs)
        # gae, actor_features = self.base(inputs)
        # dist = self.dist(actor_features)

        value = self.value_net(inputs)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        gae, _ = self.base(inputs, action)

        return gae, value, action, action_log_probs

    def get_value(self, inputs):
        value = self.value_net(inputs)
        return value

    def evaluate_actions(self, inputs, action):
        dist, gae, actor_features = self.run_actor(inputs)
        # gae, actor_features = self.base(inputs, action)
        # dist = self.dist(actor_features)

        value = self.value_net(inputs)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return actor_features, gae, value, action_log_probs, dist_entropy

class SpatialSoftmax2d(nn.Module):
    def __init__(self):
        super(SpatialSoftmax2d, self).__init__()

    def forward(self, input):
        return F.softmax(
            input.reshape(input.shape[0], input.shape[1], input.shape[2] * input.shape[3]),
            dim=2).reshape(input.shape)

class VINnet(IDAACnet):
    """
    VIN Network
    """
    def __init__(self, obs_shape, num_actions, base_kwargs=None, num_planning_steps=10):
        super(VINnet, self).__init__(obs_shape, num_actions, base_kwargs=base_kwargs)

        self.num_planning_steps = num_planning_steps
        self.num_actions = num_actions

        in_channels = 1
        hidden_channels = 16

        self.reward_model = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, padding=0), 
            nn.ReLU(inplace=True), 
            nn.Conv2d(hidden_channels, 1, kernel_size=1, padding=0), 
        )
        self.transition_model = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True), 
            nn.Conv2d(hidden_channels, 1, kernel_size=3, padding=1), 
        )
        self.done_model = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, padding=0), 
            nn.ReLU(inplace=True), 
            nn.Conv2d(hidden_channels, 1, kernel_size=1, padding=0), 
        )
        self.attention_model = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True), 
            nn.Conv2d(hidden_channels, 1, kernel_size=3, padding=1), 
            SpatialSoftmax2d()
        )
        self.dynamics_model = nn.Conv2d(2, num_actions, kernel_size=3, padding=1)

    def run_actor(self, x):
        gae, actor_features = self.base(x)

        # Reshape to be 2d
        in_representation = torch.reshape(actor_features, (actor_features.shape[0], 1, 16, 16))

        # Generate VIN Components
        reward_grid = self.reward_model(in_representation)
        transition_info = self.transition_model(in_representation)
        done_grid = self.done_model(in_representation)
        attention_grid = self.attention_model(in_representation)

        # Do VIN
        value = torch.zeros_like(reward_grid)
        for _ in range(self.num_planning_steps):
            stacked = torch.cat([value, reward_grid], dim=1)
            assert stacked.shape[0] == value.shape[0]
            assert stacked.shape[1] == 2 * value.shape[1]
            assert stacked.shape[2] == value.shape[2]
            assert stacked.shape[3] == value.shape[3]

            q_grid = self.dynamics_model(stacked * transition_info)
            assert q_grid.shape[0] == value.shape[0]
            assert q_grid.shape[1] == self.num_actions
            assert q_grid.shape[2] == value.shape[2]
            assert q_grid.shape[3] == value.shape[3]

            value_grid, _ = torch.max(q_grid, dim=1, keepdim=True)
            assert value_grid.shape[0] == value.shape[0]
            assert value_grid.shape[1] == 1
            assert value_grid.shape[2] == value.shape[2]
            assert value_grid.shape[3] == value.shape[3]

            value = done_grid * value_grid
            assert value.shape[0] == value.shape[0]
            assert value.shape[1] == 1
            assert value.shape[2] == value.shape[2]
            assert value.shape[3] == value.shape[3]


        # Apply attention model to select cell
        q_outputs = torch.sum((attention_grid * q_grid), dim=[2, 3])

        assert q_outputs.shape[0] == value.shape[0]
        assert q_outputs.shape[1] == self.num_actions

        dist = FixedCategorical(logits=q_outputs)

        return dist, gae, actor_features