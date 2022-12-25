import os
import random
import shutil
from typing import Tuple, Dict
from multiprocessing import Pool

import torch
from torch import nn

from .RL.env import MahjongGBEnv
from .feature import FeatureAgent
from .RL.model_pool import ModelPoolClient


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
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(planes)
        if stride > 1 or inplanes != planes:
            self.downsample = nn.Conv2d(inplanes, planes, 1, stride)
        else:
            self.downsample = None

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(identity)
        out += identity
        out = self.relu(out)

        return out


class CNNModel(nn.Module):

    def __init__(self, verbose=False):
        self.verbose = verbose
        nn.Module.__init__(self)
        self._embed = nn.Linear(4 * 9, 64)

        self._block1 = BasicBlock(FeatureAgent.OBS_SIZE, 256, 1)
        self._block2 = BasicBlock(256, 256, 1)
        self._conv3 = nn.Conv2d(256, 32, 3, 1, 1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self._logits = nn.Sequential(
            nn.Linear(32*8*8, 256),
            nn.ReLU(True),
            nn.Linear(256, FeatureAgent.ACT_SIZE)
        )

        self._value_branch = nn.Sequential(
            nn.Linear(32*8*8, 256),
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
        hidden = self._block1(embed)  # (batch, obs_size, 8, 8) -> (batch, 256, 4, 4)
        hidden = self._block2(hidden)  # (batch, 256, 4, 4) -> (batch, 512, 2, 2)
        hidden = self._conv3(hidden)
        # hidden = self.avgpool(hidden)  # (batch, 512, 2, 2) -> (batch, 512, 1, 1)
        # hidden = torch.squeeze(hidden)  # (batch, 512,)
        hidden = hidden.flatten(1)

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
    def __init__(self, model_dir='model/checkpoint', verbose=False):
        self.verbose = verbose
        self.model_dir = model_dir
        
    def get_model(self, name = '') -> CNNModel:
        model = CNNModel()
        if len(name):
            model.load_state_dict(torch.load(os.path.join(self.model_dir,  name)))
        return model

    def get_best_model(self, n_episode=10):
        """Have competition among candidate checkpoints in self.model_dir.
        We use knockout competition to select the best model."""
        candidates = []
        for _, _, file_list in os.walk(self.model_dir):
            for file_name in file_list:
                if '.pt' in file_name:
                    candidates.append(file_name)
        random.shuffle(candidates)
        # We move the failed models to another directory for easy recovery.
        loser_dir = self.model_dir + '_lose'
        os.makedirs(loser_dir, exist_ok=True)
        # The comparison details will be written in the log.
        logger = open(os.path.join(loser_dir, 'log.txt'), mode='a')

        def find_best_player(r):
            best_p = None
            best_r = 0
            for v in r.values():
                if best_p is None or v[1] > best_r:
                    best_p = [v[0]]
                    best_r = v[1]
                elif v[1] == best_r:
                    best_p.append(v[0])
            return best_p, best_r
        
        round = 0
        while len(candidates) >= 4:
            winners = []  # Winner for the current round.
            print (f'----------------ROUND {round}, {len(candidates)} candidates-----------------')
            round += 1
            with Pool(4) as p:
                args = []
                for idx, i in enumerate(range(0, len(candidates), 4)): 
                    if i + 3 >= len(candidates): break 
                    args.append((
                    candidates[i], candidates[i + 1], candidates[i + 2], candidates[i + 3], n_episode))
                results = p.starmap(self.compare_models, args)
            for arg, res in zip(args, results):
                best_player, best_reward = find_best_player(res)
                print(
                    f'Match between {arg[0]}, {arg[1]}, {arg[2]}, {arg[3]},'
                    f'winner: {best_player}, accumulated reward: {best_reward}')
                logger.write(
                    f'Match between {arg[0]}, {arg[1]}, {arg[2]}, {arg[3]},'
                    f'winner: {best_player}, accumulated reward: {best_reward}\n')
                if len(best_player) > 2:
                    # If more than 2 players have the same score, we randomly select one to enter the next round.
                    best_player = [best_player[0]]
                winners.extend(best_player)
                for j in range(4):
                    if arg[j] not in best_player:
                        shutil.move(
                            os.path.join(self.model_dir, arg[j]), os.path.join(loser_dir, arg[j]))
            random.shuffle(winners)
            candidates = winners  # Winners in this round will enter the next round competition.
        final_winner = None
        # Deal with the remaining candidates (less than 4).
        if len(candidates) == 2:
            results = self.compare_models(candidates[0], candidates[1], candidates[0], candidates[1], n_episode=10)
            best_player, _ = find_best_player(results)
            final_winner = best_player[0]
        elif len(candidates) == 3:
            results = self.compare_models(candidates[0], candidates[1], candidates[2], candidates[0], n_episode=10)
            best_player, _ = find_best_player(results)
            final_winner = best_player[0]
        elif len(candidates) == 1:
            final_winner = candidates[0]
        print(f'Compare {candidates}, the final winner is {final_winner}!')
        logger.write(f'Compare {candidates}, the final winner is {final_winner}!\n')
        for c in candidates:
            if c != final_winner:
                shutil.move(os.path.join(self.model_dir, c), os.path.join(loser_dir, c))

        return final_winner


    def compare_models(self, candidate1, candidate2, candidate3, candidate4, n_episode=10) -> Dict[str, int]:
        """Compare four models by playing mahjong for n_episode rounds.
        Return the accumulated reward for each player.
            results =
                {0: [candidate1, reward1], 1: [candidate2, reward2], 2: [candidate3, reward3], 3: [candidate4, reward4]}
        """
        policies = {f'player_{i+1}': CNNModel() for i in range(4)}
        candidates = [candidate1, candidate2, candidate3, candidate4]

        if self.verbose:
            print(f"comparing {candidate1} {candidate2} {candidate3} {candidate4}")
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        for player, candidate in zip(policies, candidates):
            policies[player].load_state_dict(
                torch.load(os.path.join(self.model_dir, candidate), map_location=device))
            policies[player].train(False)  # Batch Norm inference mode
            policies[player].to(device)
        
        rewards = self._compare_models(policies, n_episode, 0)

        results = {k: [candidate, rewards[f'player_{k+1}']] for k, candidate in enumerate(candidates)}

        return results
    
    def compare_baseline_latest(self, model_pool_name):
        model_pool = ModelPoolClient(model_pool_name)
        win = True
        overall_rank = 0
        for _ in range(5):
            new_player = f'player_{random.randint(1,4)}'
            policies = {f'player_{i}':CNNModel() for i in range(1,5)}
            for i, player in enumerate(policies):
                if i == new_player: 
                    latest = model_pool.get_latest_model()
                    state_dict = model_pool.load_model(latest)
                    policies[player].load_state_dict(state_dict)
                else:
                    policies[player].load_state_dict(model_pool.get_baseline_model())
                policies[player].train(False)
            results = self._compare_models(policies, 10)
            rank = 0
            for player in policies:
                if player != new_player and results[player] > results[new_player]:
                    rank += 1
            baseline_rewards = max([results[player] for player in policies if player != new_player])
            our_reward = results[new_player]
            print('reward', results, new_player, 'our_rank', rank)
            overall_rank += rank + 1
        return overall_rank < 1.9
    
    def _compare_models(self, policies, n_episode=10, idx = 0) -> Dict[str, int]:
        """Compare four models by playing mahjong for n_episode rounds.
        Return the accumulated reward for each player.
            results =
                {0: [candidate1, reward1], 1: [candidate2, reward2], 2: [candidate3, reward3], 3: [candidate4, reward4]}
        """
        env = MahjongGBEnv(config={'agent_clz': FeatureAgent})
        results = {name: 0 for name in env.agent_names}
        
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        for player in policies:
            policies[player].train(False)  # Batch Norm inference mode
            policies[player].to(device)
        
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
                    state['observation'] = torch.tensor(state['observation'], dtype=torch.float).unsqueeze(0).to(device)
                    state['action_mask'] = torch.tensor(state['action_mask'], dtype=torch.float).unsqueeze(0).to(device)
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

            for player in rewards:
                results[player] += rewards[player]
            if self.verbose:
                print(f'Episode {episode}: n_step = {n_step}, rewards = {rewards}')
        for k in results:
            results[k] /= n_episode
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
            print (f'[Manager]: load {model_path}')
        latest_version += 1
        return model, latest_version

    def get_botzone_model(self, model_dir='/data'):
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
            print("No Module!")  # botzone使用简单交互：异常可以print，正常不能print
        return model

    def save(self, model, version):
        path = self.model_dir + f'/model_{version}.pt'
        torch.save(model.state_dict(), path)
        print(f'[Manager]: save {version} to {path}')
