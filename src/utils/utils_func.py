import torch
import os
import pickle
import pandas as pd
import random
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.nist_score import corpus_nist


"""
common utils
"""
def load_dataset(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def txt_read(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    lines = [l.strip() for l in lines]
    return lines


def txt_write(path, data):
    with open(path, 'w') as f:
        f.writelines(data)


def save_data(base_path):
    if not (os.path.isfile(base_path+'data/processed/data.train') and os.path.isfile(base_path+'data/processed/data.val') and os.path.isfile(base_path+'data/processed/data.test')):
        print('Processing the chabot data')
        raw_data_path = base_path + 'data/raw/conversation_data_01.csv'
        df = pd.read_csv(raw_data_path)
        src, trg = df.iloc[:, 0].tolist(), df.iloc[:, 1].tolist()
        assert len(src) == len(trg)

        tmp = []
        for s, t in zip(src, trg):
            tmp += [s + '\n']
            tmp += [t + '\n']

        with open(base_path+'data/raw/all_data.txt', 'w') as f:
            f.writelines(tmp)
        
        random.seed(999)
        all_id = list(range(len(src)))
        tmp = random.sample(all_id, 2000)
        testset_id = random.sample(tmp, 1000)
        valset_id = list(set(tmp) - set(testset_id))
        trainset_id = list(set(all_id) - set(tmp))
        id_list = [trainset_id, valset_id, testset_id]

        for split, ids in zip(['train', 'val', 'test'], id_list):
            save_path = base_path + 'data/processed/data.' + split
            tmp = [(src[id], trg[id]) for id in ids]

            with open(save_path, 'wb') as f:
                pickle.dump(tmp, f)
                

def make_dataset_path(base_path):
    dataset_path = {}
    for split in ['train', 'val', 'test']:
        dataset_path[split] = base_path+'data/processed/data.'+split
    return dataset_path


def save_checkpoint(file, model, optimizer):
    state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
    torch.save(state, file)
    print('model pt file is being saved\n')
    

def bleu_score(ref, pred, weights):
    smoothing = SmoothingFunction().method3
    return corpus_bleu(ref, pred, weights, smoothing)


def nist_score(ref, pred, n):
    return corpus_nist(ref, pred, n)


def cal_scores(ref, pred, type, n_gram):
    assert type in ['bleu', 'nist']
    if type == 'bleu':
        wts = tuple([1/n_gram]*n_gram)
        return bleu_score(ref, pred, wts)
    return nist_score(ref, pred, n_gram)


def tensor2list(src, ref, pred, tokenizer):
    src, ref, pred = torch.cat(src, dim=0)[:, 1:], torch.cat(ref, dim=0)[:, 1:], torch.cat(pred, dim=0)[:, :-1]
    src = [tokenizer.tokenize(tokenizer.decode(src[i].tolist())) for i in range(src.size(0))]
    ref = [[tokenizer.tokenize(tokenizer.decode(ref[i].tolist()))] for i in range(ref.size(0))]
    pred = [tokenizer.tokenize(tokenizer.decode(pred[i].tolist())) for i in range(pred.size(0))]
    return src, ref, pred



def print_samples(src, ref, pred, ids):
    print('-'*50)
    for i in ids:
        s, r, p = ' '.join(src[i]), ' '.join(ref[i][0]), ' '.join(pred[i])
        print('src : {}'.format(s))
        print('gt  : {}'.format(r))
        print('pred: {}\n'.format(p))
    print('-'*50 + '\n')