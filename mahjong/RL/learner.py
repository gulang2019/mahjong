import time
from multiprocessing import Process

import torch
from torch.nn import functional as F
import os

from mahjong import CNNModel, ModelManager
from .model_pool import ModelPoolServer


class Learner(Process):

    def __init__(self, config, replay_buffer, verbose = False):
        self.verbose = verbose
        super(Learner, self).__init__()
        self.replay_buffer = replay_buffer
        self.config = config
        self.losses = []
        self.manager = ModelManager(self.config['ckpt_save_path'])

    def run(self):
        # create model pool
        model_pool = ModelPoolServer(self.config['model_pool_size'], self.config['model_pool_name'])

        # initialize model params
        device = torch.device(self.config['device'])
        # model = CNNModel(verbose = True)
        model, version = self.manager.get_latest_model(verbose = True)
        '''
        Support continuous trainining
        '''

        # send to model pool
        model_pool.push(model.state_dict())  # push cpu-only tensor to model_pool
        model = model.to(device)

        # training
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config['lr'])

        # wait for initial samples
        while self.replay_buffer.size() < self.config['min_sample']:
            time.sleep(0.1)

        cur_time = time.time()
        iterations = 0
        while True:
            # sample batch
            batch = self.replay_buffer.sample(self.config['batch_size'])
            obs = torch.tensor(batch['state']['observation']).to(device)
            mask = torch.tensor(batch['state']['action_mask']).to(device)
            states = {
                'observation': obs,
                'action_mask': mask
            }
            actions = torch.tensor(batch['action']).unsqueeze(-1).to(device)
            advs = torch.tensor(batch['adv']).to(device)
            targets = torch.tensor(batch['target']).to(device)

            

            # calculate PPO loss
            model.train(True)  # Batch Norm training mode
            old_logits, _ = model(states)
            if torch.sum(torch.isnan(old_logits)): continue
            old_probs = F.softmax(old_logits, dim=1).gather(1, actions)
            old_log_probs = torch.log(old_probs).detach()
            for _ in range(self.config['epochs']):
                logits, values = model(states)
                try:
                    action_dist = torch.distributions.Categorical(logits=logits)
                except:
                    print ('error logits', logits)
                    raise RuntimeError
                probs = F.softmax(logits, dim=1).gather(1, actions)
                log_probs = torch.log(probs)
                ratio = torch.exp(log_probs - old_log_probs)
                surr1 = ratio * advs
                surr2 = torch.clamp(ratio, 1 - self.config['clip'], 1 + self.config['clip']) * advs
                policy_loss = -torch.mean(torch.min(surr1, surr2))
                value_loss = torch.mean(F.mse_loss(values.squeeze(-1), targets))
                entropy_loss = -torch.mean(action_dist.entropy())
                loss = policy_loss + self.config['value_coeff'] * value_loss + self.config[
                    'entropy_coeff'] * entropy_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            print('Iteration %d, replay buffer in %d out %d loss %f' % (
            iterations, self.replay_buffer.stats['sample_in'], self.replay_buffer.stats['sample_out'], loss))
            self.losses.append(loss)
            
            # push new model
            model = model.to('cpu')
            model_pool.push(model.state_dict())  # push cpu-only tensor to model_pool
            model = model.to(device)

            # save checkpoints
            t = time.time()
            if t - cur_time > self.config['ckpt_save_interval']:
                self.manager.save(model, version)
                version += 1
                cur_time = t
            iterations += 1
