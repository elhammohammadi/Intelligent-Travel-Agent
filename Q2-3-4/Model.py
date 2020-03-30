import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

from tqdm import tqdm_notebook

from scipy.special import softmax

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import sys
sys.path.append('../Tools')
from Metrics import short_report

import warnings
warnings.filterwarnings("ignore")

###################################

class Model(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_type, n_hidden, n_layers, n_out, dropout, emb_dropout, use_country):
        super().__init__()
        
        '''Instantiate model parameters.'''
        
        self.vocab_size, self.embedding_dim, self.hidden_type, self.n_hidden, self.n_layers, self.n_out, self.dropout, self.emb_dropout, self.use_country = \
            vocab_size, emb_dim, hidden_type, n_hidden, n_layers, n_out, dropout, emb_dropout, use_country
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.embedding_dropout = nn.Dropout(p=self.emb_dropout)
        
        if self.hidden_type == 'RNN':
            self.hidden = nn.RNN(input_size=self.embedding_dim, hidden_size=self.n_hidden, num_layers=self.n_layers, dropout = self.dropout)
        elif self.hidden_type == 'GRU':
            self.hidden = nn.GRU(input_size=self.embedding_dim, hidden_size=self.n_hidden, num_layers=self.n_layers, dropout = self.dropout)
        elif self.hidden_type == 'LSTM':
            self.hidden = nn.LSTM(input_size=self.embedding_dim, hidden_size=self.n_hidden, num_layers=self.n_layers, dropout = self.dropout)
        
        if self.use_country == True:
            self.country_embedding = nn.Embedding(7, 10)
            self.linear = nn.Linear(self.n_hidden+10, 100)

        else:
            self.linear = nn.Linear(self.n_hidden, 100)

        self.dropout1 = nn.Dropout(p=self.dropout)
        self.relu = nn.ReLU()
        self.dropout2 = nn.Dropout(p=self.dropout)
        self.out = nn.Linear(100, self.n_out)

        
    def forward(self, X, lengths, mask_tensor, country_id, hidden_state=None):
        
        '''Process the input sequence, and output the predictions.'''
        
        bs = X.size(1)
        embs = self.embedding(X)
        
        embs = pack_padded_sequence(embs, lengths)
        if hidden_state is None:
            self.h = Variable(torch.zeros(self.n_layers, bs, self.n_hidden))
            self.c = Variable(torch.zeros(self.n_layers, bs, self.n_hidden))
        else:
            self.h, self.c = hidden_state
        if next(self.parameters()).is_cuda == True:
            self.h, self.c = self.h.cuda(), self.c.cuda()
            
        if self.hidden_type == 'RNN' or self.hidden_type == 'GRU':
            hidden_out, self.h = self.hidden(embs, self.h)
        elif self.hidden_type == 'LSTM':
            hidden_out, (self.h, self.c) = self.hidden(embs, (self.h, self.c))

        hidden_out, lengths = pad_packed_sequence(hidden_out)
        
        if self.use_country == True:
            country_ids = torch.cat([country_id.unsqueeze(0) for _ in range(lengths[0])], dim=0)
            countries = self.country_embedding(country_ids)
            hidden_out = torch.cat((hidden_out, countries), -1)
            outp = self.out(self.dropout2(self.relu(self.linear(self.dropout1(hidden_out)))))
        else:
            outp = self.out(self.dropout2(self.relu(self.linear(self.dropout1(hidden_out)))))

        outp.data[mask_tensor == 1] = -np.inf

        return outp, (self.h, self.c)


    def fit(self, train_dl, dev_dl, class_weight=None, epochs=100, learning_rate=1e-3, weight_decay=1e-2, clipping=True, verbose=True):
        
        '''The function to train the model.'''
        
        if next(self.parameters()).is_cuda == True and class_weight is not None:
            class_weight = class_weight.cuda()

        opt = optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), learning_rate, weight_decay=weight_decay)
        loss_fn=nn.CrossEntropyLoss(weight=class_weight)

        max_f1 = 0
        best_state_dict = None

        train_loss_values, dev_loss_values = [], []
        
        for epoch in tqdm_notebook(range(epochs)):

            num_batch = len(train_dl)
            y_true_train = list()
            y_pred_train = list()
            y_prob_train = list()
            total_loss_train = 0
            
            y_true_dev = list()
            y_pred_dev = list()
            y_prob_dev = list()
            total_loss_dev = 0

            for m in self.modules():
                m = m.train()

            t = tqdm_notebook(iter(train_dl), leave=False, total=num_batch)
            for user_id, country_id, (X, lengths), (y, _), mask_tensor in t:
                t.set_description(f'Epoch {epoch+1}')

                if next(self.parameters()).is_cuda == True:
                    X = Variable(X).cuda()
                    y = Variable(y).cuda()
                    country_id = Variable(country_id).cuda()
                    mask_tensor = mask_tensor.cuda()
                else:
                    X = Variable(X)
                    y = Variable(y)
                    country_id = Variable(country_id)

                lengths = lengths.numpy()

                opt.zero_grad()
                pred, (_, _) = self.forward(X, lengths, mask_tensor, country_id)
                pred = torch.cat([pred[:lengths[i], i] for i in range(len(lengths))], dim=0).view(-1, self.vocab_size)
                y = torch.cat([y[:lengths[i], i] for i in range(len(lengths))], dim=0).view(-1)
                loss = loss_fn(pred, y)
                loss.backward()
                if clipping == True:
                    clip_grad_norm_(self.parameters(), max_norm=5)
                
                opt.step()
                
                t.set_postfix(loss=loss.data)
                pred_idx = torch.max(pred[:, 2:], dim=1)[1]
                y_true_train.append(y.cpu().data.numpy()-2)
                y_pred_train.append(pred_idx.cpu().data.numpy())
                y_prob_train.append(softmax(pred[:, 2:].cpu().data.numpy(), axis=1))
                total_loss_train += loss.data
            
            y_true_train = np.concatenate(y_true_train, axis=0)
            y_pred_train = np.concatenate(y_pred_train, axis=0)
            y_prob_train = np.concatenate(y_prob_train, axis=0)

            train_report = short_report(y_true_train, y_pred_train, y_prob_train)
            train_loss = total_loss_train/len(train_dl)

            train_loss_values.append(train_loss)

            self.eval()

            for user_id, country_id, (X, lengths), (y, _), mask_tensor in tqdm_notebook(dev_dl, leave = False):
          
                if next(self.parameters()).is_cuda == True:
                    X = Variable(X).cuda()
                    y = Variable(y).cuda()
                    country_id = Variable(country_id).cuda()
                    mask_tensor = mask_tensor.cuda()
                else:
                    X = Variable(X)
                    y = Variable(y)
                    country_id = Variable(country_id)

                lengths = lengths.numpy()
                  
                pred, (_, _) = self.forward(X, lengths, mask_tensor, country_id)
                pred = torch.cat([pred[:lengths[i], i] for i in range(len(lengths))], dim=0).view(-1, self.vocab_size)
                y = torch.cat([y[:lengths[i], i] for i in range(len(lengths))], dim=0).view(-1)
                loss = loss_fn(pred, y)
                pred_idx = torch.max(pred[:, 2:], dim=1)[1]
                y_true_dev.append(y.cpu().data.numpy()-2)
                y_pred_dev.append(pred_idx.cpu().data.numpy())
                y_prob_dev.append(softmax(pred[:, 2:].cpu().data.numpy(), axis=1))
                total_loss_dev += loss.data
            
            y_true_dev = np.concatenate(y_true_dev, axis=0)
            y_pred_dev = np.concatenate(y_pred_dev, axis=0)
            y_prob_dev = np.concatenate(y_prob_dev, axis=0)

            dev_report = short_report(y_true_dev, y_pred_dev, y_prob_dev)
            dev_loss = total_loss_dev/len(dev_dl)

            dev_loss_values.append(dev_loss)
            
            report_string = '-----------------------------------------------------------\n'
            report_string += 'Epoch = %d\n\nTrain:\n' % (epoch+1)
            report_string += train_report + '\n\n'
            report_string += 'Dev:\n' + dev_report + '\n\n'
            
            if verbose == True:
                print(report_string)

                plt.plot(np.array(train_loss_values), color='b')
                plt.plot(np.array(dev_loss_values), color='r')
                plt.show()

            new_f1 = f1_score(y_true_dev, y_pred_dev, average='macro')
            if new_f1 > max_f1:
                max_f1 = new_f1
                best_state_dict = self.state_dict()

        if best_state_dict is not None:
            self.load_state_dict(best_state_dict)