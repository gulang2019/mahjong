import torch
from torch import nn

from .RL.env import MahjongGBEnv
from .feature import FeatureAgent
from typing import Tuple, Dict
import os

class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            inplanes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        if stride > 1:
            self.downsample = nn.Conv2d(inplanes, planes, 1, stride)
        else:
            self.downsample = None

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        if self.downsample is not None:
            identity = self.downsample(identity)
        out += identity
        out = self.relu(out)

        return out


class CNNModel(nn.Module):

    def __init__(self, verbose = False):
        self.verbose = verbose
        nn.Module.__init__(self)
        self._embed = nn.Linear(4*9, 64)

        self._block1 = BasicBlock(FeatureAgent.OBS_SIZE, 256, 2)
        self._block2 = BasicBlock(256, 512, 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self._logits = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Linear(256, FeatureAgent.ACT_SIZE)
        )

        self._value_branch = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Linear(256, 1)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, input_dict):
        self.train(mode=input_dict.get("is_training", False))
        obs = input_dict["observation"].float()
        embed = self._embed(obs)
        embed = embed.reshape(-1, FeatureAgent.OBS_SIZE, 8, 8)  # (batch, obs_size, 4*9) -> (batch, obs_size, 8, 8)
        hidden = self._block1(embed)            # (batch, obs_size, 8, 8) -> (batch, 256, 4, 4)
        hidden = self._block2(hidden)           # (batch, 256, 4, 4) -> (batch, 512, 2, 2)
        hidden = self.avgpool(hidden)           # (batch, 512, 2, 2) -> (batch, 512, 1, 1)
        hidden = torch.squeeze(hidden)          # (batch, 512,)

        logits = self._logits(hidden)
        mask = input_dict["action_mask"].float()
        # if self.verbose:
        #     print("mask", mask)
        inf_mask = torch.clamp(torch.log(mask), -1e38, 1e38)
        # if self.verbose:
        #     print("inf_mask", inf_mask)
        masked_logits = logits + inf_mask
        # if self.verbose:
        #     print("masked_logits", masked_logits)
        value = self._value_branch(hidden)
        return masked_logits, value


'''
model with version v and score s is stored at model_dir/model_{v}.pt
'''

class ModelManager:
    def __init__(self, model_dir = 'model/checkpoint'):
        self.model_dir = model_dir

    def get_model(self, *args, **kwargs) -> Tuple[CNNModel, int]:
        return CNNModel(*args, **kwargs)
    
    def get_best_model(self, candidate1, candidate2, candidate3, candidate4, n_episode=10) -> Dict[str, int]:
        env = MahjongGBEnv(config={'agent_clz': FeatureAgent})
        policies = {player: CNNModel() for player in env.agent_names}
        results = {candidate1: 0, candidate2: 0, candidate3: 0, candidate4: 0}
        player2ckpt = {}
        for player, candidate_ckpt in zip(policies.keys(), results.keys()):
            policies[player].load_state_dict(
                torch.load(os.path.join(self.model_dir, candidate_ckpt), map_location='cpu'))
            policies[player].train(False)  # Batch Norm inference mode
            player2ckpt[player] = candidate_ckpt

        for episode in range(n_episode):
            obs = env.reset()
            done = False
            n_step = 0
            while not done:
                # each player take action
                actions = {}
                values = {}
                for agent_name in obs:
                    state = obs[agent_name]
                    state['observation'] = torch.tensor(state['observation'], dtype=torch.float).unsqueeze(0)
                    state['action_mask'] = torch.tensor(state['action_mask'], dtype=torch.float).unsqueeze(0)
                    with torch.no_grad():
                        logits, value = policies[agent_name](state)
                        action_dist = torch.distributions.Categorical(logits=logits)
                        action = action_dist.sample().item()
                        value = value.item()
                    actions[agent_name] = action
                    values[agent_name] = value
                # interact with env
                next_obs, rewards, done = env.step(actions)
                obs = next_obs
                n_step += 1

            best_reward = 0
            best_reward_player = []
            for agent_name in rewards:
                if len(best_reward_player) is None or rewards[agent_name] > best_reward:
                    best_reward = rewards[agent_name]
                    best_reward_player = [agent_name]
                elif rewards[agent_name] == best_reward:
                    best_reward_player.append(agent_name)
            print(f'Episode {episode}: n_step = {n_step}, winner = {best_reward_player}, winner reward = {best_reward}')

        return results

    def get_latest_model(self, *args, **kwargs) -> Tuple[CNNModel, int]:
        model = CNNModel(*args, **kwargs)
        latest_version = -1
        for file in os.listdir(self.model_dir):
            if 'model' in file:
                version = int(file.split('.')[0].split('_')[1])
                if version > latest_version:
                    latest_version = version
        if latest_version != -1:
            model_path = f'{self.model_dir}/model_{latest_version}.pt'
            model.load_state_dict(torch.load(model_path))
            print (f'[Manaer]: load {model_path}')
        latest_version += 1
        return model, latest_version

    def get_botzone_model(self, model_dir = '/data'):
        self.model_dir = model_dir
        model = CNNModel()
        latest_version = -1
        for file in os.listdir(self.model_dir):
            if 'model' in file:
                version = int(file.split('.')[0].split('_')[1])
                if version > latest_version:
                    latest_version = version
        if latest_version != -1:
            model_path = f'{self.model_dir}/model_{latest_version}.pt'
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        else:
            print("No Module!")     # botzone使用简单交互：异常可以print，正常不能print
        return model

    def save(self, model, version):
        path = self.model_dir + f'/model_{version}.pt'
        torch.save(model.state_dict(), path)
        print (f'[Manager]: save {version} to {path}')
