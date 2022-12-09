from multiprocessing import Process

import numpy as np
import torch

from mahjong import FeatureAgent
from mahjong import CNNModel, ModelManager
from .env import MahjongGBEnv
from .model_pool import ModelPoolClient
import random

class Actor(Process):

    pos = 0

    def __init__(self, config, replay_buffer):
        super(Actor, self).__init__()
        self.replay_buffer = replay_buffer
        self.config = config
        self.name = config.get('name', 'Actor-?')
        self.manager = ModelManager()
        self.position = Actor.pos % 4
        Actor.pos += 1
        
    def run(self):
        torch.set_num_threads(1)

        # connect to model pool
        model_pool = ModelPoolClient(self.config['model_pool_name'])

        # create network model
        model = CNNModel()

        # load initial model
        version = model_pool.get_latest_model()
        state_dict = model_pool.load_model(version)
        model.load_state_dict(state_dict)
        
        # the best model after supervised training
        supervised_model = self.manager.get_model('model_4.pt')
        
        # collect data
        env = MahjongGBEnv(config={'agent_clz': FeatureAgent})
        policies = {}
        for i, player in enumerate(env.agent_names):
            if i == self.position: 
                policies[player] = model 
            else:
                policies[player] = supervised_model

        for episode in range(self.config['episodes_per_actor']):
            # update model
            latest = model_pool.get_latest_model()
            if latest['id'] > version['id']:
                state_dict = model_pool.load_model(latest)
                model.load_state_dict(state_dict)
                version = latest

            # run one episode and collect data
            obs = env.reset()
            episode_data = {agent_name: {
                'state': {
                    'observation': [],
                    'action_mask': []
                },
                'action': [],
                'reward': [],
                'value': []
            } for agent_name in env.agent_names}
            done = False
            n_step = 0
            while not done:
                # each player take action
                actions = {}
                values = {}
                for agent_name in obs:
                    agent_data = episode_data[agent_name]
                    state = obs[agent_name]
                    agent_data['state']['observation'].append(state['observation'])
                    agent_data['state']['action_mask'].append(state['action_mask'])
                    state['observation'] = torch.tensor(state['observation'], dtype=torch.float).unsqueeze(0)
                    state['action_mask'] = torch.tensor(state['action_mask'], dtype=torch.float).unsqueeze(0)
                    model.train(False)  # Batch Norm inference mode
                    with torch.no_grad():
                        logits, value = model(state)
                        action_dist = torch.distributions.Categorical(logits=logits)
                        action = action_dist.sample().item()
                        value = value.item()
                    actions[agent_name] = action
                    values[agent_name] = value
                    agent_data['action'].append(actions[agent_name])
                    agent_data['value'].append(values[agent_name])
                # interact with env
                next_obs, rewards, done = env.step(actions)
                for agent_name in rewards:
                    episode_data[agent_name]['reward'].append(rewards[agent_name])
                obs = next_obs
                n_step += 1
            print(self.name, 'pos', self.position, 'Episode', episode, 'Model', latest['id'], 'Reward', rewards[f'player_{self.position}'], 'Step', n_step)

            no_winner = True
            for agent_name in rewards:
                no_winner = no_winner and rewards[agent_name]
            
            if no_winner and random.random() < 0.8:
                continue
            
            # postprocessing episode data for each agent
            for agent_name, agent_data in episode_data.items():
                if len(agent_data['action']) < len(agent_data['reward']):
                    agent_data['reward'].pop(0)
                obs = np.stack(agent_data['state']['observation'])
                # print ("obs.shape", obs.shape, len(agent_data['state']['observation']))
                mask = np.stack(agent_data['state']['action_mask'])
                # print ("mask.shape", obs.shape, len(agent_data['state']['action_mask']))
                actions = np.array(agent_data['action'], dtype=np.int64)
                rewards = np.array(agent_data['reward'], dtype=np.float32)
                values = np.array(agent_data['value'], dtype=np.float32)
                next_values = np.array(agent_data['value'][1:] + [0], dtype=np.float32)

                td_target = rewards + next_values * self.config['gamma']
                td_delta = td_target - values
                advs = []
                adv = 0
                for delta in td_delta[::-1]:
                    adv = self.config['gamma'] * self.config['lambda'] * adv + delta
                    advs.append(adv)  # GAE
                advs.reverse()
                advantages = np.array(advs, dtype=np.float32)
                
                # print ("obs.shape", obs.shape)
                # print ("mask.shape", mask.shape)
                # print ("actions.shape", actions.shape)
                # print ("advantages.shape", advantages.shape)
                # print ("target.shape", td_target.shape)
                
                # send samples to replay_buffer (per agent)
                self.replay_buffer.push({
                    'state': {
                        'observation': obs,
                        'action_mask': mask
                    },
                    'action': actions,
                    'adv': advantages,
                    'target': td_target
                })
