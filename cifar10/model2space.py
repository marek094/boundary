import torch
import torchvision
from pathlib import Path
from model_mcnn import CNN
from train import get_hparams, datasets
import struct

def token_to_flags(token):
    hparams = get_hparams()
    flags = {}
    for opt in token.split(':'):
        name, type_ = next((x[1][2:], x[2]) for x in hparams if x[0][1:] == opt[:1])
        flags[name] = type_(opt[1:])
    return flags


def save_space_binary(model, dataloader, path, path_eval=None, verbose=True):
    model.eval()
    is_first = True
    criterion = torch.nn.CrossEntropyLoss()
    test_loss, correct, total = 0, 0, 0
    with path.open('wb') as f:
        for batch, (inputs, targets) in enumerate(dataloader):
            feat_batch = model._features(inputs)
            for clss, feat in zip(targets, feat_batch):
                feat = feat.tolist()
                if is_first:
                    f.write(struct.pack('<i', len(feat)))
                    is_first = False
                f.write(struct.pack('<i', clss.tolist()))
                f.write(struct.pack(f'<{len(feat)}f', *feat))

            outputs = model.fc(feat_batch)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if verbose:
                print(f'Batch {batch} saved.', end='\r')

    valid_loss, valid_acc = test_loss / total, 100. * correct / total
    if path_eval is not None:
        with path_eval[1].open('w') as fcsv:
            epoch = path_eval[0]
            print(epoch, valid_loss, valid_acc, sep=',', file=fcsv)   
    if verbose:
        print("Test epoch saved" + "--- " * 12)


def main(args):

    args.output.mkdir(exist_ok=True, parents=True)
        
    validset = datasets()['valid']
    validloader = torch.utils.data.DataLoader(
        validset, 
        batch_size=100, 
        shuffle=False)


    for model_path in args.model:
        assert model_path.exists()

        _, token, trial, epoch = model_path.stem.split('_')
        flags = token_to_flags(token)

        model = CNN(flags['width'], num_classes=10)
        model.load_state_dict(torch.load(model_path))

        result = args.output / f'space_{token}_{trial}_{epoch}.dat'
        csv = args.output / f'valid_{token}_{trial}_{epoch}.csv'
        save_space_binary(model, validloader, result, path_eval=(epoch, csv), verbose=args.verbose)



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, required=True, nargs='+')
    parser.add_argument('--output', type=Path, default='spaces')
    parser.add_argument('--verbose', '-v', action='store_true')

    args = parser.parse_args()
    main(args)
