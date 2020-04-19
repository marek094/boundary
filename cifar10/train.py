import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import os, time, csv, math

from torch.optim.lr_scheduler import StepLR, LambdaLR
from pathlib import Path

from utils import *
from model_resnet18k import make_resnet18k
from model_mcnn import CNN


def get_hparams():
    return [
        ['-x', '--name',   str,   "test"],
        ['-w', '--wdecay', float, 5e-4],
        ['-l', '--lrate',  float, 0.1],
        ['-L', '--ldecay', int,   1000],
        ['-n', '--width',  int,   10],
        ['-s', '--noise',  int,   15],
        ['-o', '--optim',  str,   "adam"],
        ['-m', '--model',  str,   "resnet18"],
    ]


def get_arguments():
    return [
        ['-t', '--trials', int,   5],
        ['-e', '--epochs', int,   1000],
        [None, '--logdir', Path,  Path('out')],
        [None, '--gpu',    int,   -1],
        [None, '--saveall',int,   10],
        [None, '--pers',   int,   -1],
    ]


def save_feature_space(model, dataloader, path, cuda=True, verbose=True):
    import struct
    is_first = True
    with path.open('wb') as f:
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            if cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            feat_batch = model._features(inputs)
            for clss, feat in zip(targets, feat_batch):
                feat = feat.tolist()
                if is_first:
                    f.write(struct.pack('<i', len(feat)))
                    is_first = False
                f.write(struct.pack('<i', clss.tolist()))
                f.write(struct.pack(f'<{len(feat)}f', *feat))
            if verbose:
                print(f'Batch {batch_idx} saved.', end='\r')
    if verbose:
        print("Test epoch saved" + "--- " * 12)


def build_parser():
    import argparse
    parser = argparse.ArgumentParser()
    hparams, arguments = get_hparams(), get_arguments()
    for line in hparams+arguments:
        if len(line) == 2:
            parser.add_argument(*filter(None, line[:2]), action='store_true')
        else:
            parser.add_argument(*filter(None, line[:2]), type=line[-2], default=line[-1])

    return parser


def settings_token(flags_dict):
    hparams = get_hparams()
    token = ":".join([f'{short[1:]}{flags_dict[name[2:]]}' for short, name, _, _ in hparams])
    return token


def datasets():
    transform_train = transforms.Compose([
        # transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(),
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
        if flags.gpu >= 0:
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return train_loss / total, 100. * correct / total


def valid(model, validloader, criterion, flags):
    model.eval()
    test_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(validloader):
            if flags.gpu >= 0:
                inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    return test_loss / total, 100. * correct / total


def main(flags):
    # directory
    token = settings_token(vars(flags))
    timestamp = int(time.time())
    logdir = flags.logdir / token
    logdir.mkdir(exist_ok=True, parents=True)

    # dataset
    sets = datasets()
    trainset, validset = sets['train'], sets['valid']
    validloader = torch.utils.data.DataLoader(validset, batch_size=100, shuffle=False)

    for trial in range(1, flags.trials+1):
        noise_size = len(trainset) * flags.noise // 100
        trainsubset = get_dataset_label_noise(trainset, noise_size=noise_size)
        trainloader = torch.utils.data.DataLoader(trainsubset, batch_size=128, shuffle=True)

        # model & loss
        criterion = nn.CrossEntropyLoss()
        if flags.model == 'resnet18':
            model = make_resnet18k(flags.width, num_classes=10)
        elif flags.model == 'cnn':
            model = CNN(flags.width, num_classes=10)

        # move to GPU
        if flags.gpu >= 0:
            criterion = criterion.cuda()
            model = model.cuda()
        
        if flags.optim == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=flags.lrate, weight_decay=flags.wdecay)
            scheduler = StepLR(optimizer, step_size=flags.ldecay, gamma=0.1)
        elif flags.optim == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=flags.lrate, momentum=0.9, weight_decay=flags.wdecay)
            trainset_size = 50_000
            lr_lambda = lambda epoch: flags.lrate / math.sqrt(1 + (epoch*trainset_size)//512)
            scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda, last_epoch=-1)

        best_acc, best_epoch = 0, -1
        for epoch in range(1, flags.epochs+1):
            train_loss, train_acc = train(model, trainloader, optimizer, criterion, flags)
            valid_loss, valid_acc = valid(model, validloader,            criterion, flags)
            scheduler.step(epoch)

            with open(logdir / f'trainlog_{token}_{trial}_{epoch}_{timestamp}.csv', 'w') as csvf:
                print(epoch, train_loss, train_acc, valid_loss, valid_acc, sep=',', file=csvf)
            
            if flags.pers >= 0 and (epoch == 1 or epoch % flags.saveall == 0):
                dat = logdir / f'space_{token}_{trial}_{epoch}.dat'
                # pers = logdir / f'space_{token}_{trial}_{epoch}.ph.txt'
                save_feature_space(model, validloader, dat, cuda=(flags.gpu>=0))
                # os.system(f'../persistence/vcomplex {dat} |' +
                #           f'../persistence/ripser/ripser --dim {flags.pers} --threshold 999000 ' + 
                #           f' > {pers}'
                #          )
                # dat.unlink()
                continue

            if epoch % flags.saveall == 0:
                torch.save(model.state_dict(), logdir / f'model_{token}_{trial}_{epoch}.pkl')
            
            if epoch >= 10 and valid_acc > best_acc:
                prev = logdir / f'model_{token}_{trial}_{best_epoch}.pkl'
                if prev.exists() and best_epoch % flags.saveall > 0:
                    prev.unlink()
                torch.save(model.state_dict(), logdir / f'model_{token}_{trial}_{epoch}.pkl')
                best_acc, best_epoch = valid_acc, epoch
            if epoch == flags.epochs:
                torch.save(model.state_dict(), logdir / f'model_{token}_{trial}_{epoch}.pkl')



if __name__ == "__main__":
    import os
    parser = build_parser()
    args = parser.parse_args()
    cuda = "" if args.gpu < 0 else args.gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = f'{cuda}'
    main(args)