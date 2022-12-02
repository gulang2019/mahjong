# Model part

# 修改：
# 类型：Tensor->torch.Tensor
# 重复代码：obs = input_dict["observation"].float()删去
# 对齐：obs = obs.reshape(-1, 133, 1, 36)
# 第一维是batch不展开：hidden = torch.flatten(hidden, 1)          # (512,)

import torch
from torch import nn
from feature import FeatureAgent


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

    def __init__(self):
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

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, input_dict):
        self.train(mode=input_dict.get("is_training", False))
        obs = input_dict["obs"]["observation"].float()
        # obs = input_dict["observation"].float()                   ?
        obs = obs.reshape(-1, 133, 1, 36)
        embed = self._embed(obs).reshape(FeatureAgent.OBS_SIZE, 8, 8)  # (obs_size, 4*9) -> (obs_size, 8, 8)
        hidden = self._block1(embed)            # (obs_size, 8, 8) -> (256, 4, 4)
        hidden = self._block2(hidden)           # (256, 4, 4) -> (512, 2, 2)
        hidden = self.avgpool(hidden)           # (512, 2, 2) -> (512, 1, 1)
        hidden = torch.flatten(hidden, 1)          # (512,)
        action_logits = self._logits(hidden)
        action_mask = input_dict["obs"]["action_mask"].float()
        inf_mask = torch.clamp(torch.log(action_mask), -1e38, 1e38)
        return action_logits + inf_mask
