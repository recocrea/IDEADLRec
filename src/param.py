import os
import yaml
import pprint
import argparse

def parse_args(parse=True, **optional_kwargs):
    parser = argparse.ArgumentParser()

    #system args
    parser.add_argument('--data_dir', default='../data/', type=str)
    parser.add_argument('--output_dir', default='output/', type=str)
    parser.add_argument('--data_name', default='Beauty', type=str)
    parser.add_argument('--do_eval', default=False)

    # model args
    parser.add_argument("--visual_size", type=int, default=512, help="hidden size of transformer model")
    parser.add_argument("--hidden_size", type=int, default=128, help="hidden size of transformer model")
    parser.add_argument('--num_attention_heads', default=2, type=int)
    parser.add_argument('--hidden_act', default="gelu", type=str)
    parser.add_argument("--initializer_range", type=float, default=0.02)
    parser.add_argument('--max_seq_length', default=50, type=int)
    parser.add_argument('--use_fuse', default=True, type=bool)
    parser.add_argument('--use_contrastive', default=False, type=bool)
    parser.add_argument('--eps', default=1e-12, type=float)
    parser.add_argument("--num_layers", type=int, default=1, help="number of layers")
    parser.add_argument('--nv', default=4, type=int)
    parser.add_argument('--nh', default=16, type=int)
    parser.add_argument('--reg_weight', default=1e-4, type=float)
    parser.add_argument('--lmd', default=0.1, type=float)
    parser.add_argument('--tau', default=1, type=float)
    parser.add_argument('--sim', default='dot', type=str)

    parser.add_argument('--model_idx', default='ablation', type=str)
    parser.add_argument("--model_name", default='SASRec', type=str)
    parser.add_argument("--dropout_prob", type=float, default=0.5, help="dropout p")

    # train args
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate of adam")
    parser.add_argument("--batch_size", type=int, default=256, help="number of batch_size")
    parser.add_argument("--epochs", type=int, default=300, help="number of epochs")
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--log_freq", type=int, default=1, help="per epoch print res")
    parser.add_argument("--seed", default=42, type=int)

    #learning related
    parser.add_argument("--weight_decay", type=float, default=0.0, help="weight_decay of adam")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam second beta value")

    if parse:
        args = parser.parse_args()
    else:
        args = parser.parse_known_args()[0]

    args.output_dir = os.path.join(args.output_dir, args.model_name)

    args.data_file = args.data_dir + args.data_name + '.txt'
    args.visual_file = args.data_dir + args.data_name + '.pt'

    args_str = f'{args.model_name}-{args.data_name}-{args.model_idx}-{args.hidden_size}'
    args.log_file = os.path.join(args.output_dir, args_str + '.txt')

    checkpoint = args_str + '.pt'
    args.checkpoint_path = os.path.join(args.output_dir, checkpoint)

    args.args_str = args_str

    kwargs = vars(args)
    kwargs.update(optional_kwargs)

    args = Config(**kwargs)
    args.save(os.path.join(args.output_dir, args_str+'.yaml'))

    return args

class Config(object):
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    @property
    def config_str(self):
        return pprint.pformat(self.__dict__)

    def __repr__(self):
        config_str = 'Configurations\n'
        config_str += self.config_str
        return config_str

    def save(self, path):
        with open(path, 'w') as f:
            yaml.dump(self.__dict__, f, default_flow_style=False)

    @classmethod
    def load(cls, path):
        with open(path, 'r') as f:
            kwargs = yaml.full_load(f)
        return Config(**kwargs)
