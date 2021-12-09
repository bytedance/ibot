import torch
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--checkpoint_backbone', required=True, type=str)
parser.add_argument('--checkpoint_linear', required=True, type=str)
parser.add_argument('--output_file', required=True, type=str)

if __name__ == "__main__":
    args = parser.parse_args()
    backbone = torch.load(args.checkpoint_backbone)['state_dict']
    model = torch.load(args.checkpoint_linear)
    linear = model['state_dict']
    head_index = model['best_acc_hidx']

    new_linear = {}
    for key, val in linear.items():
        splits = key.split('.')
        if splits[0] == str(head_index):
            new_linear['.'.join(splits[2:])] = val
    backbone.update(new_linear)
    model['state_dict'] = backbone
    
    print(f"save {head_index}th head with acc {model['best_acc']}")
    torch.save(model, args.output_file)