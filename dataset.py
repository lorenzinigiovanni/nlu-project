import os
from io import open
import torch


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, device):
        self.dictionary = Dictionary()
        self.device = device

        # add the word <pad> to the dictionary to have it in position 0
        self.dictionary.add_word('<pad>')

        self.train = self.tokenize(os.path.join('ptbdataset', 'ptb.train.txt'))
        self.valid = self.tokenize(os.path.join('ptbdataset', 'ptb.valid.txt'))
        self.test = self.tokenize(os.path.join('ptbdataset', 'ptb.test.txt'))

    def tokenize(self, path):
        # read the dataset files and store the tokens in a dictionary
        with open(path, 'r') as f:
            sentences = []
            for line in f:
                # add the start of sentence and the end of sentence to the sentence read from the file
                words = ['<sos>'] + line.split() + ['<eos>']
                # list to store the sentence
                sentence = []

                for word in words:
                    # add every word to the dictionary
                    self.dictionary.add_word(word)
                    # convert every word to it's corresponding ID and store it in a list
                    sentence.append(self.dictionary.word2idx[word])

                # create a tensor starting from the list and append it to a list of sentences
                sentences.append(torch.tensor(sentence, device=self.device, dtype=torch.int64))

        return sentences
