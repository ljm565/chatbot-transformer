import os
import random
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset
from utils import LOGGER, colorstr
from utils.filesys_utils import txt_write, write_dataset



def seed_worker(worker_id):  # noqa
    """Set dataloader worker seed https://pytorch.org/docs/stable/notes/randomness.html#dataloader."""
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def preprocess_data(config):
    raw_data_dir = os.path.join(config.counselor_dataset.path, 'counselor/raw')
    procssed_data_dir = os.path.join(config.counselor_dataset.path, 'counselor/processed')
    os.makedirs(procssed_data_dir, exist_ok=True)
    
    if not all([os.path.isfile(os.path.join(procssed_data_dir, f'data.{s}')) for s in ['train', 'val', 'test']]):
        LOGGER.info(colorstr('Processing the chabot data'))
        raw_data_path = os.path.join(raw_data_dir, 'conversation_data_01.csv')
        df = pd.read_csv(raw_data_path)
        src, trg = df.iloc[:, 0].tolist(), df.iloc[:, 1].tolist()
        assert len(src) == len(trg)

        tmp = []
        for s, t in zip(src, trg):
            tmp += [s + '\n']
            tmp += [t + '\n']

        txt_write(os.path.join(raw_data_dir, 'all_data.txt'), tmp)
        
        all_id = list(range(len(src)))
        tmp = random.sample(all_id, 2000)
        testset_id = random.sample(tmp, 1000)
        valset_id = list(set(tmp) - set(testset_id))
        trainset_id = list(set(all_id) - set(tmp))
        id_list = [trainset_id, valset_id, testset_id]

        for split, ids in zip(['train', 'val', 'test'], id_list):
            save_path = os.path.join(procssed_data_dir, f'data.{split}')
            tmp = [(src[id], trg[id]) for id in ids]
            write_dataset(save_path, tmp)

    dataset_paths = {s: os.path.join(procssed_data_dir, f'data.{s}') for s in ['train', 'val', 'test']}
    dataset_paths['validation'] = dataset_paths.pop('val')      # change 'val' to 'validaiton'

    return dataset_paths


class DLoader(Dataset):
    def __init__(self, data, tokenizer, config):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = config.max_len
        self.length = len(self.data)

    def add_special_token(self, s, tokenizer):
        s = [tokenizer.bos_token_id] + tokenizer.encode(s)[:self.max_len-2] + [tokenizer.eos_token_id]
        s = s + [tokenizer.pad_token_id] * (self.max_len - len(s))
        return s

    def __getitem__(self, idx):
        src, trg = self.add_special_token(self.data[idx][0], self.tokenizer), self.add_special_token(self.data[idx][1], self.tokenizer)
        return torch.LongTensor(src), torch.LongTensor(trg)

    def __len__(self):
        return self.length