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


def save_space_binary(model, dataloader, path, verbose=True):
    model.eval()
    is_first = True
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
            if verbose:
                print(f'Batch {batch} saved.', end='\r')
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
        save_space_binary(model, validloader, result, verbose=args.verbose)



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, required=True, nargs='+')
    parser.add_argument('--output', type=Path, default='spaces')
    parser.add_argument('--verbose', '-v', action='store_true')

    args = parser.parse_args()
    main(args)
