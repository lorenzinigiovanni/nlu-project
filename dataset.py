import os
from io import open
import torch
from torch._C import dtype


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, device):
        self.dictionary = Dictionary()
        self.device = device

        self.dictionary.add_word('<pad>')

        # self.train = self.tokenize(os.path.join('ptbdataset', 'ptb.train.txt'))
        # self.valid = self.tokenize(os.path.join('ptbdataset', 'ptb.valid.txt'))
        # self.test = self.tokenize(os.path.join('ptbdataset', 'ptb.test.txt'))

        self.train = self.engoppenize(os.path.join('ptbdataset', 'ptb.train.txt'))
        self.valid = self.engoppenize(os.path.join('ptbdataset', 'ptb.valid.txt'))
        self.test = self.engoppenize(os.path.join('ptbdataset', 'ptb.test.txt'))

    def tokenize(self, path):
        with open(path, 'r') as f:
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    self.dictionary.add_word(word)

        with open(path, 'r') as f:
            idss = []
            for line in f:
                words = line.split() + ['<eos>']
                ids = []
                for word in words:
                    ids.append(self.dictionary.word2idx[word])
                idss.append(torch.tensor(ids).type(torch.int64))
            ids = torch.cat(idss)

        return ids

    def engoppenize(self, path):
        with open(path, 'r') as f:
            for line in f:
                words = ['<sos>'] + line.split() + ['<eos>']
                for word in words:
                    self.dictionary.add_word(word)

        with open(path, 'r') as f:
            idss = []
            for line in f:
                words = ['<sos>'] + line.split() + ['<eos>']
                ids = []
                for word in words:
                    ids.append(self.dictionary.word2idx[word])
                idss.append(torch.tensor(ids, device=self.device).type(torch.int64))

        return idss
