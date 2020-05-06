#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2020 LeonTao
#
# Distributed under terms of the MIT license.

"""
Train POS models
"""

import sys
import time
import argparse

from tqdm import tqdm
import numpy as np

import torch
import torch.optim as optim

from dataset import build_vocab
from dataset import get_iterators
from config import VocabConfig, ModelConfig, TrainConfig

from lstmcrf import LSTMCRF

parser = argparse.ArgumentParser(sys.argv[0], description='build vocab by passing pos train file. (each line contains a char and a processed tag)')
parser.add_argument('input_file', help='processed input file, each line has two columns, one is char and another is processed tag')

args = parser.parse_args()

train_iterator, valid_iterator, test_iterator = get_iterators(args.input_file)
print('train_iterator: ', train_iterator)

build_vocab(train_iterator.dataset)

src_vocab = train_iterator.dataset.fields['TEXT'].vocab
#src_vocab.load_vectors()
#src_embedding = src_vector
tgt_vocab = train_iterator.dataset.fields['TAG'].vocab

src_size = len(src_vocab.stoi)
tgt_size = len(tgt_vocab.stoi)

print('src_size: ', src_size)
print('tgt_size: ', tgt_size)
print('src pad: ', src_vocab.stoi[VocabConfig.PAD])
print('tgt pad: ', tgt_vocab.stoi[VocabConfig.PAD])

def build_model():
    if TrainConfig.model == 'lstm':
        model = LSTMCRF(src_size, tgt_size,
                        tgt_vocab.stoi[VocabConfig.PAD], ModelConfig)
    elif TrainConfig.model == 'bert':
        pass
    return model

model = build_model().to(TrainConfig.device)
print(model)

optimizer = torch.optim.Adam(
    model.parameters(),
    TrainConfig.lr,
    betas=(0.9, 0.98),
    eps=1e-09
)


def epochs():
    for epoch in range(1, TrainConfig.epochs + 1):
        print('[ Epoch', epoch, ']')

        start = time.time()
        train_loss = train_epoch(epoch)

        print(' (Training)   loss: {ppl: 8.5f}, elapse: {elapse:3.3f} min'.format(
                    loss=train_loss,
                    elapse=(time.time()-start)/60))

        start = time.time()
        valid_accu = valid_epoch(epoch)
        print(' (Validation) accuracy: {accu:3.3f} %, '
                'elapse: {elapse:3.3f} min'.format(
                    accu=100*valid_accu,
                    elapse=(time.time()-start)/60)
                )

def train_epoch(epoch):
    model.train()

    total_loss = 0

    for i, batch in tqdm(enumerate(train_iterator), mininterval=2,
            desc=' (Training: %d) ' % epoch, leave=False):

        optimizer.zero_grad()

        (inputs, inputs_length), tgts = batch
        #print('inputs: ', inputs.shape)
        #print(inputs)
        #print('inputs_length: ', inputs_length.shape)
        #print('tgts: ', tgts.shape)
        inputs_mask = torch.arange(0, inputs.shape[0]).long() \
							.repeat(TrainConfig.batch_size, 1) \
							.lt(inputs_length.unsqueeze(1)) \
                            .T \
							.to(TrainConfig.device)
        #print('inputs_mask: ', inputs_mask.shape)
        #print(inputs_mask)

        loss = -model(inputs, inputs_length, inputs_mask, tgts)
        #print('loss: ', loss)

        if i % 50 == 0:
            print('loss: ', loss.item())

        total_loss += loss.item()

        loss.backward()

         # update parameters
        optimizer.step()

    return total_loss / len(train_iterator)

def valid_epoch():
    model.eval()

    with torch.no_grad():
        for batch in tqdm(valid_iterator, mininterval=2,
                desc=' (Validation: %d) ' % epoch, leave=False):
            pass


if __name__ == '__main__':
    epochs()


