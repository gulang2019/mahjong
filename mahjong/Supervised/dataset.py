from bisect import bisect_right

import numpy as np
from torch.utils.data import Dataset


class MahjongGBDataset(Dataset):

    def __init__(self, begin=0, end=1, augment=False):
        import json
        with open('new_data/count.json') as f:
            self.match_samples = json.load(f)
        self.total_matches = len(self.match_samples)
        self.total_samples = sum(self.match_samples)
        self.begin = int(begin * self.total_matches)
        self.end = int(end * self.total_matches)
        self.match_samples = self.match_samples[self.begin: self.end]
        self.matches = len(self.match_samples)
        self.samples = sum(self.match_samples)
        self.augment = augment
        t = 0
        print (f'{self.matches} matches, {self.samples} samples')
        for i in range(self.matches):
            a = self.match_samples[i]
            self.match_samples[i] = t
            t += a
        self.cache = {'obs': [], 'mask': [], 'act': []}
        for i in range(self.matches):
            if i % 128 == 0: print('loading', i)
            d = np.load('data/%d.npz' % (i + self.begin))
            for k in d:
                self.cache[k].append(d[k])

    def __len__(self):
        return self.samples

    def __getitem__(self, index):
        match_id = bisect_right(self.match_samples, index, 0, self.matches) - 1
        sample_id = index - self.match_samples[match_id]
        return self.cache['obs'][match_id][sample_id], self.cache['mask'][match_id][sample_id], \
               self.cache['act'][match_id][sample_id]

class MyMahjongGBDataset(Dataset):

    def __init__(self, indices, samples):
        self.match_samples = samples
        self.total_matches = len(self.match_samples)
        self.total_samples = sum(self.match_samples)
        self.matches = len(self.match_samples)
        self.samples = sum(self.match_samples)
        self.indices = indices
        t = 0
        print (f'{self.matches} matches, {self.samples} samples')
        for i in range(self.matches):
            a = self.match_samples[i]
            self.match_samples[i] = t
            t += a
        self.cache = {'obs': [None] * self.matches, 'mask': [None] * self.matches, 'act': [None] * self.matches}
        # for i in range(self.matches):
        #     if i % 128 == 0: print('loading', i)
        #     d = np.load('new_data/%d.npz' % (indices[i]))
        #     for k in d:
        #         self.cache[k].append(d[k])

    def __len__(self):
        return self.samples

    def __getitem__(self, index):
        match_id = bisect_right(self.match_samples, index, 0, self.matches) - 1
        sample_id = index - self.match_samples[match_id]
        if self.cache['obs'][match_id] is None:
            d = np.load('new_data/%d.npz' % (self.indices[match_id]))
            for k in d:
                self.cache[k][match_id] = d[k]
        return self.cache['obs'][match_id][sample_id], self.cache['mask'][match_id][sample_id], \
               self.cache['act'][match_id][sample_id]
import math
import random
def train_validation_split(train_ratio = 0.9, n_sample=math.inf):
    import json
    train_indices = []
    train_samples = []
    validation_indices = []
    validation_samples = []
    with open('new_data/count.json') as f:
        match_samples = json.load(f)
    for i, sample in enumerate(match_samples):
        if i >= n_sample:
            break
        if random.random() < train_ratio:
            train_indices.append(i)
            train_samples.append(sample)
        else:
            validation_indices.append(i)
            validation_samples.append(sample)
    print (f'train: {len(train_indices)}, validation: {len(validation_indices)}')
    return MyMahjongGBDataset(train_indices, train_samples), MyMahjongGBDataset(validation_indices, validation_samples)
