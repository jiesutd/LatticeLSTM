# -*- coding: utf-8 -*-
# @Author: Jie Yang
# @Date:   2017-10-17 16:47:32
# @Last Modified by:   Jie Yang,     Contact: jieynlp@gmail.com
# @Last Modified time: 2018-05-03 21:58:36
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
# from kblayer import GazLayer
from .charbilstm import CharBiLSTM
from .charcnn import CharCNN
from .latticelstm import LatticeLSTM

class BiLSTM(nn.Module):
    def __init__(self, data):
        super(BiLSTM, self).__init__()
        print("build batched bilstm...")
        self.use_bigram = data.use_bigram
        self.gpu = data.HP_gpu
        self.use_char = data.HP_use_char
        self.use_gaz = data.HP_use_gaz
        self.batch_size = data.HP_batch_size
        self.char_hidden_dim = 0
        if self.use_char:
            self.char_hidden_dim = data.HP_char_hidden_dim
            self.char_embedding_dim = data.char_emb_dim
            if data.char_features == "CNN":
                self.char_feature = CharCNN(data.char_alphabet.size(), self.char_embedding_dim, self.char_hidden_dim, data.HP_dropout, self.gpu)
            elif data.char_features == "LSTM":
                self.char_feature = CharBiLSTM(data.char_alphabet.size(), self.char_embedding_dim, self.char_hidden_dim, data.HP_dropout, self.gpu)
            else:
                print("Error char feature selection, please check parameter data.char_features (either CNN or LSTM).")
                exit(0)
        self.embedding_dim = data.word_emb_dim
        self.hidden_dim = data.HP_hidden_dim
        self.drop = nn.Dropout(data.HP_dropout)
        self.droplstm = nn.Dropout(data.HP_dropout)
        self.word_embeddings = nn.Embedding(data.word_alphabet.size(), self.embedding_dim)
        self.biword_embeddings = nn.Embedding(data.biword_alphabet.size(), data.biword_emb_dim)
        self.bilstm_flag = data.HP_bilstm
        # self.bilstm_flag = False
        self.lstm_layer = data.HP_lstm_layer
        if data.pretrain_word_embedding is not None:
            self.word_embeddings.weight.data.copy_(torch.from_numpy(data.pretrain_word_embedding))
        else:
            self.word_embeddings.weight.data.copy_(torch.from_numpy(self.random_embedding(data.word_alphabet.size(), self.embedding_dim)))
            
        if data.pretrain_biword_embedding is not None:
            self.biword_embeddings.weight.data.copy_(torch.from_numpy(data.pretrain_biword_embedding))
        else:
            self.biword_embeddings.weight.data.copy_(torch.from_numpy(self.random_embedding(data.biword_alphabet.size(), data.biword_emb_dim)))
        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        
        if self.bilstm_flag:
            lstm_hidden = data.HP_hidden_dim // 2
        else:
            lstm_hidden = data.HP_hidden_dim
        lstm_input = self.embedding_dim + self.char_hidden_dim
        if self.use_bigram:
            lstm_input += data.biword_emb_dim
        self.forward_lstm = LatticeLSTM(lstm_input, lstm_hidden, data.gaz_dropout, data.gaz_alphabet.size(), data.gaz_emb_dim, data.pretrain_gaz_embedding, True, data.HP_fix_gaz_emb, self.gpu)
        if self.bilstm_flag:
            self.backward_lstm = LatticeLSTM(lstm_input, lstm_hidden, data.gaz_dropout, data.gaz_alphabet.size(), data.gaz_emb_dim, data.pretrain_gaz_embedding, False, data.HP_fix_gaz_emb, self.gpu)
        # self.lstm = nn.LSTM(lstm_input, lstm_hidden, num_layers=self.lstm_layer, batch_first=True, bidirectional=self.bilstm_flag)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(data.HP_hidden_dim, data.label_alphabet_size)

        if self.gpu:
            self.drop = self.drop.cuda()
            self.droplstm = self.droplstm.cuda()
            self.word_embeddings = self.word_embeddings.cuda()
            self.biword_embeddings = self.biword_embeddings.cuda()
            self.forward_lstm = self.forward_lstm.cuda()
            if self.bilstm_flag:
                self.backward_lstm = self.backward_lstm.cuda()
            self.hidden2tag = self.hidden2tag.cuda()


    def random_embedding(self, vocab_size, embedding_dim):
        pretrain_emb = np.empty([vocab_size, embedding_dim])
        scale = np.sqrt(3.0 / embedding_dim)
        for index in range(vocab_size):
            pretrain_emb[index,:] = np.random.uniform(-scale, scale, [1, embedding_dim])
        return pretrain_emb


    def get_lstm_features(self, gaz_list, word_inputs, biword_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover):
        """
            input:
                word_inputs: (batch_size, sent_len)
                gaz_list:
                word_seq_lengths: list of batch_size, (batch_size,1)
                char_inputs: (batch_size*sent_len, word_length)
                char_seq_lengths: list of whole batch_size for char, (batch_size*sent_len, 1)
                char_seq_recover: variable which records the char order information, used to recover char order
            output: 
                Variable(sent_len, batch_size, hidden_dim)
        """
        batch_size = word_inputs.size(0)
        sent_len = word_inputs.size(1)
        word_embs =  self.word_embeddings(word_inputs)
        if self.use_bigram:
            biword_embs = self.biword_embeddings(biword_inputs)
            word_embs = torch.cat([word_embs, biword_embs],2)
        if self.use_char:
            ## calculate char lstm last hidden
            char_features = self.char_feature.get_last_hiddens(char_inputs, char_seq_lengths.cpu().numpy())
            char_features = char_features[char_seq_recover]
            char_features = char_features.view(batch_size,sent_len,-1)
            ## concat word and char together
            word_embs = torch.cat([word_embs, char_features], 2)
        word_embs = self.drop(word_embs)
        # packed_words = pack_padded_sequence(word_embs, word_seq_lengths.cpu().numpy(), True)
        hidden = None
        lstm_out, hidden = self.forward_lstm(word_embs, gaz_list, hidden)
        if self.bilstm_flag:
            backward_hidden = None 
            backward_lstm_out, backward_hidden = self.backward_lstm(word_embs, gaz_list, backward_hidden)
            lstm_out = torch.cat([lstm_out, backward_lstm_out],2)
        # lstm_out, _ = pad_packed_sequence(lstm_out)
        lstm_out = self.droplstm(lstm_out)
        return lstm_out



    def get_output_score(self, gaz_list,  word_inputs, biword_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover):
        lstm_out = self.get_lstm_features(gaz_list, word_inputs,biword_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover)
        ## lstm_out (batch_size, sent_len, hidden_dim)
        outputs = self.hidden2tag(lstm_out)
        return outputs
    

    def neg_log_likelihood_loss(self, gaz_list, word_inputs, biword_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, batch_label, mask):
        ## mask is not used
        batch_size = word_inputs.size(0)
        seq_len = word_inputs.size(1)
        total_word = batch_size * seq_len
        loss_function = nn.NLLLoss(ignore_index=0, size_average=False)
        outs = self.get_output_score(gaz_list, word_inputs, biword_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover)
        # outs (batch_size, seq_len, label_vocab)
        outs = outs.view(total_word, -1)
        score = F.log_softmax(outs, 1)
        loss = loss_function(score, batch_label.view(total_word))
        _, tag_seq  = torch.max(score, 1)
        tag_seq = tag_seq.view(batch_size, seq_len)
        return loss, tag_seq


    def forward(self, gaz_list, word_inputs, biword_inputs, word_seq_lengths,  char_inputs, char_seq_lengths, char_seq_recover, mask):
        
        batch_size = word_inputs.size(0)
        seq_len = word_inputs.size(1)
        total_word = batch_size * seq_len
        outs = self.get_output_score(gaz_list,  word_inputs, biword_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover)
        outs = outs.view(total_word, -1)
        _, tag_seq  = torch.max(outs, 1)
        tag_seq = tag_seq.view(batch_size, seq_len)
        ## filter padded position with zero
        decode_seq = mask.long() * tag_seq
        return decode_seq




