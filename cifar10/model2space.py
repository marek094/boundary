import torch
import torchvision
from pathlib import Path
from model_mcnn import CNN
from train import get_hparams, datasets
import struct
from utils import save_space_binary

def token_to_flags(token):
    hparams = get_hparams()
    flags = {}
    for opt in token.split(':'):
        name, type_ = next((x[1][2:], x[2]) for x in hparams if x[0][1:] == opt[:1])
        flags[name] = type_(opt[1:])
    return flags

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
