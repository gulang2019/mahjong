import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR

from mahjong.Supervised.dataset import MahjongGBDataset
from mahjong.model import CNNModel, ModelManager

if __name__ == '__main__':
    logdir = './model/'
    model_dir = './model/checkpoint'
    os.system('mkdir -p ./model/checkpoint')

    # Load dataset
    splitRatio = 0.9
    batchSize = 1024
    trainDataset = MahjongGBDataset(0, splitRatio, True)
    validateDataset = MahjongGBDataset(splitRatio, 1, False)
    loader = DataLoader(dataset=trainDataset, batch_size=batchSize, shuffle=True)
    vloader = DataLoader(dataset=validateDataset, batch_size=batchSize, shuffle=False)

    # Load model
    # model = CNNModel().to('cuda')
    manager = ModelManager()
    model, version = manager.get_latest_model()
    model = model.to('cuda')
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    lr_scheduler = ExponentialLR(optimizer=optimizer, gamma=0.9)        # lr_scheduler

    # Train and validate
    for e in range(16):
        print('Epoch', e)
        manager.save(model, version)
        version += 1
        correct = 0
        for i, d in enumerate(loader):
            input_dict = {'is_training': True, 'observation': d[0].cuda(), 'action_mask': d[1].cuda()}
            logits, _ = model(input_dict)
            loss = F.cross_entropy(logits, d[2].long().cuda())
            if i % 128 == 0:
                print('Iteration %d/%d' % (i, len(trainDataset) // batchSize + 1), 'policy_loss', loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pred = logits.argmax(dim=1)                                 # acc
            correct += torch.eq(pred, d[2].cuda()).sum().item()         # acc
        lr_scheduler.step()                                             # lr_scheduler
        last_lr = lr_scheduler.get_last_lr()
        acc = correct / len(trainDataset)
        print('train acc:', acc, ', last_lr:', last_lr)                 # log
        print('Run validation:')
        correct = 0
        for i, d in enumerate(vloader):
            input_dict = {'is_training': False, 'observation': d[0].cuda(), 'action_mask': d[1].cuda()}
            with torch.no_grad():
                logits, _ = model(input_dict)
                pred = logits.argmax(dim=1)
                correct += torch.eq(pred, d[2].cuda()).sum().item()
        acc = correct / len(validateDataset)
        print('Epoch', e + 1, 'Validate acc:', acc)
