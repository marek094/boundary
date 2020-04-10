# one_hot, get_subsample_dataset, get_subsample_dataset_label_noise
# are adjusted from:
# https://github.com/yaodongyu/Rethink-BiasVariance-Tradeoff

import torch
import numpy as np
import torch.nn.init as init
from random import shuffle
import copy


def one_hot(x, K):
    return np.array(x[:, None] == np.arange(K)[None, :], dtype=int)


def get_subsample_dataset(trainset, subset):
    trainsubset = copy.deepcopy(trainset)
    trainsubset.data = [trainsubset.data[index] for index in subset]
    trainsubset.targets = [trainsubset.targets[index] for index in subset]
    return trainsubset


def get_subsample_dataset_label_noise(trainset, subset, noise_size):
    train_size = len(subset)
    trainsubset = copy.deepcopy(trainset)
    trainsubset.data = [trainsubset.data[index] for index in subset]
    trainsubset.targets = [trainsubset.targets[index] for index in subset]
    ######## shuffle
    shuffle_targets_subset = [copy.deepcopy(trainsubset.targets[idx]) for idx in range(train_size - noise_size, train_size)]    
    shuffle(shuffle_targets_subset)
    for idx in range(train_size - noise_size, train_size):
        trainsubset.targets[idx] = shuffle_targets_subset[idx - train_size + noise_size]
    return trainsubset


def get_dataset_label_noise(trainset, noise_size):
    ######## shuffle
    train_size = len(trainset)
    shuffle_targets_set = [copy.deepcopy(trainset.targets[idx]) for idx in range(train_size - noise_size, train_size)]    
    shuffle(shuffle_targets_set)
    for idx in range(train_size - noise_size, train_size):
        trainset.targets[idx] = shuffle_targets_set[idx - train_size + noise_size]
    return trainset
    

def save_feature_space(model, dataloader, path, cuda=True, verbose=True):
    import struct
    is_first = True
    with path.open('wb') as f:
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            if cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            feat_batch = model.to_feature_space(inputs)
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