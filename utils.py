import os
import random

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


PAD_WORD="<pad>"
SOS_WORD="<sos>"
EOS_WORD="<eos>"
UNK_WORD="<unk>"

PAD_IDX=0
SOS_IDX=1
EOS_IDX=2
UNK_IDX=3


class LabelSmoothingLoss(nn.Module):
    """
    With label smoothing,
    KL-divergence between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w) is minimized.
    """
    def __init__(self, label_smoothing, tgt_vocab_size, ignore_index=-100):
        assert 0.0 < label_smoothing <= 1.0
        self.padding_idx = ignore_index
        super(LabelSmoothingLoss, self).__init__()

        smoothing_value = label_smoothing / (tgt_vocab_size - 2)
        one_hot = torch.full((tgt_vocab_size,), smoothing_value)
        one_hot[self.padding_idx] = 0
        self.register_buffer('one_hot', one_hot.unsqueeze(0))

        self.confidence = 1.0 - label_smoothing

    def forward(self, output, target):
        """
        output (FloatTensor): batch_size x n_classes
        target (LongTensor): batch_size
        """
        model_prob = self.one_hot.repeat(target.size(0), 1)
        model_prob.scatter_(1, target.unsqueeze(1), self.confidence)
        model_prob.masked_fill_((target == self.padding_idx).unsqueeze(1), 0)

        return F.kl_div(output, model_prob, reduction='sum')


class Dictionary(object):
    def __init__(self, word2idx=None):
        if word2idx is None:
            self.word2idx = {}
            self.idx2word = {}
            self.word2idx[PAD_WORD] = PAD_IDX
            self.word2idx[SOS_WORD] = SOS_IDX
            self.word2idx[EOS_WORD] = EOS_IDX
            self.word2idx[UNK_WORD] = UNK_IDX
            self.wordcounts = {}
        else:
            self.word2idx = word2idx
            self.idx2word = {v: k for k, v in word2idx.items()}

    # to track word counts
    def add_word(self, word):
        if word not in self.wordcounts:
            self.wordcounts[word] = 1
        else:
            self.wordcounts[word] += 1

    # prune vocab based on count k cutoff or most frequently seen k words
    def prune_vocab(self, k=5, cnt=False):
        # get all words and their respective counts
        vocab_list = [(word, count) for word, count in self.wordcounts.items()]
        if cnt:
            # prune by count
            self.pruned_vocab = \
                    {pair[0]: pair[1] for pair in vocab_list if pair[1] > k}
        else:
            # prune by most frequently seen words
            vocab_list.sort(key=lambda x: (x[1], x[0]), reverse=True)
            k = min(k, len(vocab_list))
            self.pruned_vocab = [pair[0] for pair in vocab_list[:k]]
        # sort to make vocabulary determistic
        self.pruned_vocab.sort()

        # add all chosen words to new vocabulary/dict
        for word in self.pruned_vocab:
            if word not in self.word2idx:
                self.word2idx[word] = len(self.word2idx)
        print("Original vocab {}; Pruned to {}".
              format(len(self.wordcounts), len(self.word2idx)))
        self.idx2word = {v: k for k, v in self.word2idx.items()}

    def __len__(self):
        return len(self.word2idx)


class Corpus(object):
    def __init__(self, datafiles, maxlen, vocab_size=11000, lowercase=False, vocab=None, debug=False):
        self.dictionary = Dictionary(vocab)
        self.maxlen = maxlen
        self.lowercase = lowercase
        self.vocab_size = vocab_size
        self.datafiles = datafiles
        self.forvocab = []
        self.data = {}

        if vocab is None:
            for path, name, fvocab in datafiles:
                if fvocab or debug:
                    self.forvocab.append(path)
            self.make_vocab()

        for path, name, _ in datafiles:
            self.data[name] = self.tokenize(path)


    def make_vocab(self):
        for path in self.forvocab:
            assert os.path.exists(path)
            # Add words to the dictionary
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    L = line.lower() if self.lowercase else line
                    words = L.strip().split(" ")
                    for word in words:
                        self.dictionary.add_word(word)

        # prune the vocabulary
        self.dictionary.prune_vocab(k=self.vocab_size, cnt=False)

    def tokenize(self, path):
        """Tokenizes a text file."""
        dropped = 0
        with open(path, 'r', encoding='utf-8') as f:
            linecount = 0
            lines = []
            for line in f:
                linecount += 1
                L = line.lower() if self.lowercase else line
                words = L.strip().split(" ")
                if self.maxlen > 0 and len(words) > self.maxlen:
                    dropped += 1
                    continue
                words = [SOS_WORD] + words + [EOS_WORD]
                # vectorize
                vocab = self.dictionary.word2idx
                unk_idx = vocab[UNK_WORD]
                indices = [vocab[w] if w in vocab else unk_idx for w in words]
                lines.append(indices)

        print("Number of sentences dropped from {}: {} out of {} total".
              format(path, dropped, linecount))
        return lines
    

def batchify(data, bsz, shuffle=False, gpu=False):
    if shuffle:
        random.shuffle(data)

    nbatch = len(data) // bsz
    batches = []

    for i in range(nbatch):
        # Pad batches to maximum sequence length in batch
        batch = data[i*bsz:(i+1)*bsz]
        
        # subtract 1 from lengths b/c includes BOTH starts & end symbols
        words = batch
        lengths = [len(x)-1 for x in words]

        # sort items by length (decreasing)
        batch, lengths = length_sort(batch, lengths)
        words = batch

        # source has no end symbol
        source = [x[:-1] for x in words]
        # target has no start symbol
        target = [x[1:] for x in words]

        # find length to pad to
        maxlen = max(lengths)
        for x, y in zip(source, target):
            zeros = (maxlen-len(x))*[0]
            x += zeros
            y += zeros

        source = torch.LongTensor(np.array(source).transpose(1, 0))
        target = torch.LongTensor(np.array(target).transpose(1, 0)).contiguous().view(-1)
        lengths = torch.LongTensor(np.array(lengths))

        batches.append((source, target, lengths))
    print('{} batches'.format(len(batches)))
    return batches


def length_sort(items, lengths, descending=True):
    """In order to use pytorch variable length sequence package"""
    items = list(zip(items, lengths))
    items.sort(key=lambda x: x[1], reverse=True)
    items, lengths = zip(*items)
    return list(items), list(lengths)


def truncate(words):
    # truncate sentences to first occurrence of <eos>
    truncated_sent = []
    for w in words:
        if w != EOS_WORD:
            truncated_sent.append(w)
        else:
            break
    sent = " ".join(truncated_sent)
    return sent