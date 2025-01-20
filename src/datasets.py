import torch
from utils import neg_sample
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler

# use for SASRec, GRU4Rec, FMLP, loss function is BPR
class SequentialModelDataset(Dataset):
    def __init__(self, args, user_seq, mode='train'):
        self.args = args
        self.user_seq = user_seq
        self.max_seq_length = args.max_seq_length
        self.mode = mode

    def __len__(self):
        return len(self.user_seq)

    def process_length(self, input_ids):
        length = len(input_ids)
        if length > self.max_seq_length:
            return input_ids[-self.max_seq_length:], [self.max_seq_length]
        elif length < self.max_seq_length:
            return (self.max_seq_length - length) * [0] + input_ids, [length]
        else:
            return input_ids, [length]

    def __getitem__(self, index):
        user_id = index
        items = self.user_seq[index]
        if self.mode == "train":
            input_ids, length = self.process_length(items[:-3])
            pos, _ = self.process_length(items[1:-2])
            neg = []
            for _ in input_ids:
                neg.append(neg_sample(set(items), self.args.item_size))
            answer = [items[-3]]

        elif self.mode == 'valid':
            input_ids, length = self.process_length(items[:-2])
            pos = [0]
            neg = [0]
            answer = [items[-2]]

        else:
            input_ids, length = self.process_length(items[:-1])
            pos = [0]
            neg = [0]
            answer = [items[-1]]

        cur_rec_tensors = (
            torch.tensor(user_id, dtype=torch.long),
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(length, dtype=torch.long),
            torch.tensor(pos, dtype=torch.long),
            torch.tensor(neg, dtype=torch.long),
            torch.tensor(answer, dtype=torch.long),
        )

        return cur_rec_tensors

# use for SRGNN, Caser, loss function is CrossEntropy
class SequentialDataset(Dataset):
    def __init__(self, args, user_seq, mode='train'):
        self.args = args
        self.max_seq_length = args.max_seq_length
        self.mode = mode
        self.user_seq = self.process_user_seq(user_seq)

    def __len__(self):
        return len(self.user_seq)

    def process_length(self, input_ids):
        length = len(input_ids)
        if length > self.max_seq_length:
            return input_ids[-self.max_seq_length:], self.max_seq_length
        elif length < self.max_seq_length:
            return input_ids + (self.max_seq_length - length) * [0], length
        else:
            return input_ids, length

    def process_user_seq(self, user_seq):
        ret = []
        if self.mode == "train":
            for index, seq in enumerate(user_seq):
                use_seq, length = self.process_length(seq[:-3])
                for i in range(length):
                    if i == length -1:
                        ret.append([index, use_seq, [length], [seq[-3]]])
                    else:
                        ret.append(
                            [index, use_seq[:i+1] + [0]*(self.max_seq_length - i - 1), [i+1], [use_seq[i+1]]]
                        )

        elif self.mode == "valid":
            for index, seq in enumerate(user_seq):
                input_ids, length = self.process_length(seq[:-2])
                ret.append([index, input_ids, [length], [seq[-2]]])
        else:
            for index, seq in enumerate(user_seq):
                input_ids, length = self.process_length(seq[:-1])
                ret.append([index, input_ids, [length], [seq[-1]]])
        return ret

    def __getitem__(self, index):
        user_id, input_ids, length, answer = self.user_seq[index]
        cur_rec_tensors = (
            torch.tensor(user_id, dtype=torch.long),
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(length, dtype=torch.long),
            torch.tensor(answer, dtype=torch.long),
        )
        return cur_rec_tensors


def get_loader(args, user_seq, mode='train'):

    if args.model_name.lower() in ['srgnn', 'caser']:
        rec_dataset = SequentialDataset(args, user_seq, mode=mode)
    else:
        rec_dataset = SequentialModelDataset(args, user_seq, mode=mode)

    if mode.lower() == 'train':
        sampler = RandomSampler(rec_dataset)
    else:
        sampler = SequentialSampler(rec_dataset)

    data_loader = DataLoader(
        rec_dataset,
        sampler=sampler,
        batch_size=args.batch_size,
    )

    return data_loader