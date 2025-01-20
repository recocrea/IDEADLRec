import torch
import numpy as np
from trainers import Trainer
from param import parse_args
from models import SASRecModel, GRU4Rec, Caser, FMLP, SRGNN
from datasets import get_loader
from utils import EarlyStopping, get_user_seqs, check_path, set_seed, log_write

def show_args_info(args):
    print(f"--------------------Configure Info:------------")
    for arg in vars(args):
        print(f"{arg:<30} : {getattr(args, arg):>35}")

def get_models(args):
    if args.model_name.lower() == 'sasrec':
        model = SASRecModel
    elif args.model_name.lower() == 'gru4rec':
        model = GRU4Rec
    elif args.model_name.lower() == 'caser':
        model = Caser
    elif args.model_name.lower() == 'srgnn':
        model = SRGNN
    elif args.model_name.lower() == 'fmlp':
        model = FMLP
    else:
        model = SASRecModel

    if args.do_eval:
        return model.load_model(args)
    else:
        return model(args)

def main():
    args = parse_args()
    user_seq, max_item, valid_rating_matrix, test_rating_matrix = get_user_seqs(args.data_file)
    args.item_size = max_item + 1
    args.user_size = len(user_seq) + 1

    set_seed(args.seed)
    check_path(args.output_dir)
    show_args_info(args)
    log_write(args.log_file, str(args) + '\n')

    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.train_matrix = test_rating_matrix
    # args.train_matrix = valid_rating_matrix
    train_dataloader = get_loader(args, user_seq, mode='train')
    eval_dataloader = get_loader(args, user_seq, mode='test')
    test_dataloader = get_loader(args, user_seq, mode='test')

    model = get_models(args)
    trainer = Trainer(model, train_dataloader, eval_dataloader, test_dataloader, args)

    if args.do_eval:
        trainer.args.train_matrix = test_rating_matrix
        print(f'Load model from {args.checkpoint_path} for test!')
        scores, result_info = trainer.test(0)

    else:
        print(f'Train CrossRec')
        early_stopping = EarlyStopping(args.checkpoint_path, patience=20, verbose=True)
        for epoch in range(args.epochs):
            trainer.train(epoch)
            scores, _ = trainer.valid(epoch)
            early_stopping(np.array(np.array(scores[-1:])), trainer.model)
            if early_stopping.early_stop:
                print("Early stopping")
                break
        trainer.args.train_matrix = test_rating_matrix
        print('---------------Change to test_rating_matrix!-------------------')
        trainer.model.load_state_dict(torch.load(args.checkpoint_path, weights_only=True))
        scores, result_info = trainer.test(0)

    print(args.args_str)
    print(result_info)
    log_write(args.log_file, args.args_str + '\n' + result_info + '\n')

main()
