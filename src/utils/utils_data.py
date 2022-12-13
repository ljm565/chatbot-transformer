import torch
from torch.utils.data import Dataset



class DLoader(Dataset):
    def __init__(self, data, tokenizer, config):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = config.max_len
        self.length = len(self.data)

    
    def add_special_token(self, s, tokenizer):
        s = [tokenizer.sos_token_id] + tokenizer.encode(s)[:self.max_len-2] + [tokenizer.eos_token_id]
        s = s + [tokenizer.pad_token_id] * (self.max_len - len(s))
        return s


    def __getitem__(self, idx):
        src, trg = self.add_special_token(self.data[idx][0], self.tokenizer), self.add_special_token(self.data[idx][1], self.tokenizer)
        return torch.LongTensor(src), torch.LongTensor(trg)

    
    def __len__(self):
        return self.length