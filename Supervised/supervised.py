# 改动：
# train acc
# lr_scheduler
# epoch
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR

from dataset import MahjongGBDataset
from model import CNNModel

if __name__ == '__main__':
    logdir = '/model/'
    os.mkdir(logdir + 'checkpoint')

    # Load dataset
    splitRatio = 0.9
    batchSize = 1024
    trainDataset = MahjongGBDataset(0, splitRatio, True)
    validateDataset = MahjongGBDataset(splitRatio, 1, False)
    loader = DataLoader(dataset=trainDataset, batch_size=batchSize, shuffle=True)
    vloader = DataLoader(dataset=validateDataset, batch_size=batchSize, shuffle=False)

    # Load model
    model = CNNModel().to('cuda')
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    lr_scheduler = ExponentialLR(optimizer=optimizer, gamma=0.9)        # lr_scheduler

    # Train and validate
    for e in range(30):                                                 # epoch
        print('Epoch', e)
        torch.save(model.state_dict(), logdir + 'checkpoint/%d.pkl' % e)
        correct = 0
        for i, d in enumerate(loader):
            input_dict = {'is_training': True, 'obs': {'observation': d[0].cuda(), 'action_mask': d[1].cuda()}}
            logits = model(input_dict)
            loss = F.cross_entropy(logits, d[2].long().cuda())

            pred = logits.argmax(dim=1)                                 # acc
            correct += torch.eq(pred, d[2].cuda()).sum().item()         # acc

            if i % 128 == 0:
                print('Iteration %d/%d' % (i, len(trainDataset) // batchSize + 1), 'policy_loss', loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        lr_scheduler.step()                                             # lr_scheduler
        last_lr = lr_scheduler.get_last_lr()
        acc = correct / len(trainDataset)
        print('train acc:', acc, ', last_lr:', last_lr)                 # log

        print('Run validation:')
        correct = 0
        for i, d in enumerate(vloader):
            input_dict = {'is_training': False, 'obs': {'observation': d[0].cuda(), 'action_mask': d[1].cuda()}}
            with torch.no_grad():
                logits = model(input_dict)
                pred = logits.argmax(dim=1)
                correct += torch.eq(pred, d[2].cuda()).sum().item()
        acc = correct / len(validateDataset)
        print('Epoch', e + 1, 'Validate acc:', acc)
