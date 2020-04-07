import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import time, csv

from torch.optim.lr_scheduler import StepLR
from pathlib import Path

from utils import *
from resnet18k import make_resnet18k

def get_hparams():
    return [
        ['-x', '--name',   str,   "test"],
        ['-w', '--wdecay', float, 5e-4],
        ['-l', '--lrate',  float, 0.1],
        ['-L', '--ldecay', int,   1000],
        ['-n', '--width',  int,   10],
        ['-s', '--noise',  int,   1500],
    ]


def get_arguments():
    return [
        ['-t', '--trials', int,   5],
        ['-e', '--epochs', int,   1000],
        [None, '--logdir', Path,  Path('out')],
        [None, '--gpu',    int,   -1]
    ]


def build_parser():
    import argparse
    parser = argparse.ArgumentParser()
    hparams, arguments = get_hparams(), get_arguments()
    for line in hparams+arguments:
        parser.add_argument(*filter(None, line[:2]), type=line[-2], default=line[-1])
    return parser


def settings_token(flags_dict):
    hparams = get_hparams()
    token = ":".join([f'{short[1:]}{flags_dict[name[2:]]}' for short, name, _, _ in hparams])
    return token


def datasets():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    validset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    return {'train': trainset, 'valid': validset}


def train(model, trainloader, optimizer, criterion, flags):
    model.train()
    train_loss, correct, total = 0, 0, 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        targets_onehot = torch.FloatTensor(targets.size(0), 10)
        if flags.gpu >= 0:
            inputs, targets = inputs.cuda(), targets.cuda()
            targets = targets.cuda()
        targets_onehot.zero_()
        targets_onehot.scatter_(1, targets.view(-1, 1).long(), 1)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets_onehot)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * outputs.numel()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    return train_loss / total, 100. * correct / total


def valid(model, validloader, optimizer, criterion, flags):
    model.eval()
    test_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(validloader):
            targets_onehot = torch.FloatTensor(targets.size(0), 10)
            if flags.gpu >= 0:
                inputs, targets = inputs.cuda(), targets.cuda()
                targets = targets.cuda()
            targets_onehot.zero_()
            targets_onehot.scatter_(1, targets.view(-1, 1).long(), 1)
            outputs = model(inputs)
            loss = criterion(outputs, targets_onehot)
            test_loss += loss.item() * outputs.numel()
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)
    return test_loss / total, 100. * correct / total



def main(flags):
    # directory
    token = settings_token(vars(flags))
    timestamp = time.time()
    logdir = flags.logdir / token
    logdir.mkdir(exist_ok=True, parents=True)

    # dataset
    sets = datasets()
    trainset, validset = sets['train'], sets['valid']
    # set up index after random permutation
    permute_index = np.split(np.random.permutation(len(trainset)), flags.trials)
    validloader = torch.utils.data.DataLoader(validset, batch_size=100, shuffle=False, num_workers=2)

    for trial in range(1, flags.trials+1):
        trainsubset = get_subsample_dataset_label_noise(trainset, permute_index[trial-1], noise_size=flags.noise)
        trainloader = torch.utils.data.DataLoader(trainsubset, batch_size=128, shuffle=True)

        # model & loss
        criterion = nn.MSELoss(reduction='mean')
        model = make_resnet18k(flags.width, num_classes=10)
        # move to GPU
        if args.gpu >= 0:
            criterion = criterion.cuda()
            model = model.cuda()
        
        optimizer = optim.Adam(model.parameters(), lr=flags.lrate, weight_decay=flags.wdecay)
        scheduler = StepLR(optimizer, step_size=flags.ldecay, gamma=0.1)

        with open(logdir / f'trainlog_{trial}_{timestamp}.csv', 'w') as csvf:
            best_acc, best_epoch = 0, -1
            logger = csv.DictWriter(csvf, fieldnames="epoch,train_loss,train_acc,valid_loss,valid_acc".split(','))
            logger.writeheader()
            for epoch in range(1, flags.epochs+1):
                train_loss, train_acc = train(model, trainloader, optimizer, criterion, flags)
                valid_loss, valid_acc = valid(model, validloader, optimizer, criterion, flags)
                scheduler.step(epoch)

                logger.writerow({'epoch': epoch, 
                    'train_loss': train_loss, 'train_acc': train_acc, 
                    'valid_loss': valid_loss, 'valid_acc': valid_acc,
                })
                csvf.flush()

                if epoch >= 10 and valid_acc > best_acc:
                    (logdir / f'model_{token}_{trial}_{best_epoch}.pkl').unlink(missing_ok=True)
                    torch.save(model.state_dict(), logdir / f'model_{token}_{trial}_{epoch}.pkl')
                    best_acc, best_epoch = valid_acc, epoch
                if epoch == flags.epochs:
                    torch.save(model.state_dict(), logdir / f'model_{token}_{trial}_{epoch}.pkl')



if __name__ == "__main__":
    import os
    parser = build_parser()
    args = parser.parse_args()
    cuda = "" if args.gpu < 0 else args.gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = cuda
    main(args)