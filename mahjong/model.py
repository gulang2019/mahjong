import torch
from torch import nn
from .feature import FeatureAgent
from typing import Tuple
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

    def get_best_model(self, *args, **kwargs) -> Tuple[CNNModel, int]:
        # TBD
        raise NotImplementedError

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

    def save(self, model, version):
        path = self.model_dir + f'/model_{version}.pt'
        torch.save(model.state_dict(), path)
        print (f'[Manager]: save {version} to {path}')
