import torch
import numpy as np
from trainers import Trainer
from param import parse_args
from models import SASRecModel, GRU4Rec, Caser, FMLP, SRGNN
from datasets import get_loader
from utils import EarlyStopping, get_user_seqs, check_path, set_seed, log_write



def main():
    args = parse_args()
    user_seq, max_item, valid_rating_matrix, test_rating_matrix = get_user_seqs(args.data_file)
    args.item_size = max_item + 1
    args.user_size = len(user_seq) + 1
    set_seed(args.seed)

    train_dataloader = get_loader(args, user_seq, mode='train')
    eval_dataloader = get_loader(args, user_seq, mode='test')
    test_dataloader = get_loader(args, user_seq, mode='test')


    train_data = []
    eval_data = []
    test_data = []

    for data in train_dataloader:
        train_data.append(data)
    for data in eval_dataloader:
        eval_data.append(data)
    for data in test_dataloader:
        test_data.append(data)
    torch.save(train_data, 'SequentialTrainData.pt')
    torch.save(eval_data, 'SequentialEvalData.pt')
    torch.save(test_data, 'SequentialTestData.pt')
main()
