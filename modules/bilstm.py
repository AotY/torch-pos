#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2020 LeonTao
#
# Distributed under terms of the MIT license.

"""
fork from https://github.com/allanj/pytorch_lstmcrf/blob/master/model/bilstm_encoder.py
"""
import torch
import torch.nn as nn

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class BiLSTMEncoder(nn.Module):
    def __init__(self, src_size, tgt_size, config, print_info: bool = True):
        super(BiLSTMEncoder, self).__init__()

        self.tgt_size = tgt_size

        self.embedding_dim = config.embedding_dim

        self.word_embedding = nn.Embedding(src_size, self.embedding_dim)
        # self.word_embedding = nn.Embedding.from_pretrained(torch.FloatTensor(config.word_embedding), freeze=False)

        self.word_drop = nn.Dropout(config.dropout)

        if print_info:
            print("[Model Info] Input size to LSTM: {}".format(self.embedding_dim))
            print("[Model Info] LSTM Hidden Size: {}".format(config.hidden_size))

        self.lstm = nn.LSTM(self.embedding_dim, config.hidden_size // 2, num_layers=1, batch_first=False, bidirectional=True)

        self.drop_lstm = nn.Dropout(config.dropout)

        if print_info:
            print("[Model Info] Final Hidden Size: {}".format(config.hidden_size))

        self.hidden2tgt = nn.Linear(config.hidden_size, self.tgt_size)

    def forward(self, word_seq_tensor: torch.Tensor,
                       word_seq_lens: torch.Tensor) -> torch.Tensor:
        """
        Encoding the input with BiLSTM
        :param word_seq_tensor: (batch_size, sent_len)   NOTE: The word seq actually is already ordered before come here.
        :param word_seq_lens: (batch_size, 1)
        :return: emission scores (batch_size, sent_len, hidden_size)
        """

        word_emb = self.word_embedding(word_seq_tensor)

        word_rep = self.word_drop(word_emb)

        #sorted_seq_len, permIdx = word_seq_lens.sort(0, descending=True)
        #_, recover_idx = permIdx.sort(0, descending=False)
        #sorted_seq_tensor = word_rep[permIdx]
        #packed_words = pack_padded_sequence(sorted_seq_tensor, sorted_seq_len, True)
        packed_words = pack_padded_sequence(word_rep, word_seq_lens, batch_first=False)
        lstm_out, _ = self.lstm(packed_words, None)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=False)  ## CARE: make sure here is batch_first, otherwise need to transpose.
        feature_out = self.drop_lstm(lstm_out)

        outputs = self.hidden2tgt(feature_out)

        #return outputs[recover_idx]
        return outputs # [max_len, batch_size, tgt_size]
