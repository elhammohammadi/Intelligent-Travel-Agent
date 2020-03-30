import pandas as pd
from tqdm import tnrange
import numpy as np
import os, csv, itertools

import torch
from torchtext import data
from torchtext.data import TabularDataset, BucketIterator

from sklearn.model_selection import train_test_split


class BatchWrapper:
    
    '''The final iterator class, that covers the original
       torchtext data iterator, in order to facilitate the
       access to samples in different batches.'''
    
    def __init__(self, dl, parameters, mask_dict):
        self.dl, self.parameters, self. mask_dict = dl, parameters, mask_dict

    def __iter__(self):
        for batch in self.dl:
            mask_tensor = torch.zeros(len(batch.user_id), batch.output[1][0], self.mask_dict[int(batch.user_id[0])].shape[1])
            for i, ui in enumerate(batch.user_id):
                m = self.mask_dict[int(ui)]
                mask_tensor[i, :m.shape[0], :] = torch.tensor(m)
            yield (tuple(getattr(batch, parameter) for parameter in self.parameters)) + tuple([mask_tensor.permute(1, 0, 2)])
  
    def __len__(self):
        return len(self.dl)


def read_data(path='city_search.json'):

    '''Given the original JSON data file, output a flattened dataframe.'''
    
    df = pd.read_json(path)

    nested = []
    for i in range(len(df)):
        nested.append(df.user[i][0][0])

    new_df = pd.DataFrame(nested)

    data = pd.concat([df, new_df], axis=1)

    data.loc[data['country'] == '', 'country'] = 'UNK'
    data = data.drop(['user', '_row'], axis=1)
    
    countries = sorted(list(data.country.unique()))
    countries.append(countries.pop(countries.index('UNK')))
    countries = {country: i for i, country in enumerate(countries)}
    data['country_id'] = data['country'].map(countries)
    
    for i in range(len(data)):
        data.at[i, 'cities'] = data['cities'][i][0].split(', ')
        data.at[i, 'session_id'] = data['session_id'][i][0]
        data.at[i, 'unix_timestamp'] = data['unix_timestamp'][i][0]
        data['country_id']
    
    return data


def user_builder(df):
    
    '''Given the flattened dataframe, output a user-based collection of data.'''
    
    data = []
    user_ids = df.user_id.unique()

    for i in tnrange(len(user_ids)):
        user = user_ids[i]
        sub_df = df.loc[df['user_id'] == user].reset_index(drop=True)
        country = sub_df['country_id'][0]
        join_date = sub_df['joining_date'][0]
        sorted_df = sub_df.sort_values('unix_timestamp').reset_index(drop=True)
        history = sorted_df['cities'].tolist()
        timestamps =  sorted_df['unix_timestamp'].tolist()
        history = [(timestamps[j], history[j]) for j in range(len(sorted_df))]
        data.append((user, country, join_date, history))
  
    return data


def separate_data(data):
    
    '''Use the average number of searched cities per session
       for different users, in order to make a stratified
       train-dev-test split.'''
    
    user_averages = []
    for i in range(len(data)):
        cities = []
        for session in data[i][3]:
            cities.extend(session[1])
        user_averages.append(len(cities)/len(data[i][3]))
    
    bins = np.linspace(0, len(user_averages), 20)
    y_binned = np.digitize(user_averages, bins)
    X_train, X_devtest, y_train, y_devtest = train_test_split(data, user_averages, test_size=0.30, random_state=42, stratify=y_binned)


    bins = np.linspace(0, len(y_devtest), 20)
    y_binned = np.digitize(y_devtest, bins)
    X_dev, X_test, y_dev, y_test = train_test_split(X_devtest, y_devtest, test_size=0.50, random_state=42, stratify=y_binned)
    
    return X_train, X_dev, X_test


def calculate_mask_matrix(data, vocab):
    
    '''Create a mask matrix to filter out the
       previous city/cities within the same session.'''
    
    mask_dict = {}

    for s in data:
        user_id = int(s.user_id)
        y = s.output

        mask_matrix = np.zeros((len(y), len(vocab)))
        mask_matrix[:, :2].fill(1)

        seen_set = set()
        for i, word in enumerate(y):
            if len(seen_set) == 0:
                mask_matrix[i, vocab.stoi['<sos>']] = 1
            for seen_word in seen_set:
                mask_matrix[i, vocab.stoi[seen_word]] = 1
            if word == '<sos>':
                seen_set = set()
                continue
            seen_set.add(word)

        mask_dict[user_id] = mask_matrix
    
    return mask_dict


def calculate_class_weight(train_data, vocab, power=1):
    
    '''Given the city counts, output a weight matrix
       proportional to the inverse of the number of
       cities within the training data.'''
    
    train_tokens = list(itertools.chain.from_iterable([t.output for t in train_data]))
    token_count = np.zeros((len(vocab)-2))
    for tok in train_tokens:
        token_count[vocab.stoi[tok]-2] += 1

    class_weight = (len(train_tokens) / (2 * token_count)) ** power
    class_weight = torch.FloatTensor(np.divide(class_weight, np.sum(class_weight)))
    class_weight = torch.cat([torch.FloatTensor([0, 0]), class_weight], dim=0)

    return class_weight


def prepare_data(train_samples, dev_samples, test_samples, batch_size=32, power=1/3, create_iterators=True):
    
    '''Given the train-dev-test user-based collections, create
       dataloaders, vocabulary list (list of all labels),
       alongside the class-weight tensor.'''
    
    train_data, dev_data, test_data = [], [], []

    for t in train_samples:
        x = '<sos>,'+',<sos>,'.join([','.join(s[1]) for s in t[3]])
        y = ',<sos>,'.join([','.join(s[1]) for s in t[3]])+',<sos>'
        train_data.append((t[0], t[1], t[2], x, y))
    for d in dev_samples:
        x = '<sos>,'+',<sos>,'.join([','.join(s[1]) for s in d[3]])
        y = ',<sos>,'.join([','.join(s[1]) for s in d[3]])+',<sos>'
        dev_data.append((d[0], d[1], d[2], x, y))
    for t in test_samples:
        x = '<sos>,'+',<sos>,'.join([','.join(s[1]) for s in t[3]])
        y = ',<sos>,'.join([','.join(s[1]) for s in t[3]])+',<sos>'
        test_data.append((t[0], t[1], t[2], x, y))

    with open('train.tsv', 'w', newline='\n') as file:
        csv_out=csv.writer(file, delimiter='\t')
        csv_out.writerow(['user_id', 'country_id', 'input', 'output'])
        for row in train_data:
            csv_out.writerow((row[0], row[1], row[3], row[4]))
    with open('dev.tsv', 'w', newline='\n') as file:
        csv_out=csv.writer(file, delimiter='\t')
        csv_out.writerow(['user_id', 'country_id', 'input', 'output'])
        for row in dev_data:
            csv_out.writerow((row[0], row[1], row[3], row[4]))
    with open('test.tsv', 'w', newline='\n') as file:
        csv_out=csv.writer(file, delimiter='\t')
        csv_out.writerow(['user_id', 'country_id', 'input', 'output'])
        for row in test_data:
            csv_out.writerow((row[0], row[1], row[3], row[4]))

    ID = data.Field(sequential=False, use_vocab=False)
    TEXT = data.Field(sequential=True, tokenize=lambda x: x.split(','), lower=False, include_lengths=True)

    datafields = [('user_id', ID), ('country_id', ID), ('input', TEXT), ('output', TEXT)]

    train_data = TabularDataset(path='train.tsv', format='tsv', fields=datafields, skip_header=True)
    dev_data = TabularDataset(path='dev.tsv', format='tsv', fields=datafields, skip_header=True)
    test_data = TabularDataset(path='test.tsv', format='tsv', fields=datafields, skip_header=True)

    os.remove('train.tsv')
    os.remove('dev.tsv')
    os.remove('test.tsv')

    TEXT.build_vocab(train_data.input)

    class_weight = calculate_class_weight(train_data, TEXT.vocab, power)

    train_mask = calculate_mask_matrix(train_data, TEXT.vocab)
    dev_mask = calculate_mask_matrix(dev_data, TEXT.vocab)
    test_mask = calculate_mask_matrix(test_data, TEXT.vocab)
    
    if create_iterators == False:
        return (train_data, train_mask), (dev_data, dev_mask), (test_data, test_mask), TEXT.vocab

    train_dl = BucketIterator(train_data, batch_size=batch_size, device=None, sort_key=lambda x: len(x.input), sort_within_batch=True, repeat=False)
    dev_dl = BucketIterator(dev_data, batch_size=batch_size, device=None, sort_key=lambda x: len(x.input), sort_within_batch=True, repeat=False)
    test_dl = BucketIterator(test_data, batch_size=batch_size, device=None, sort_key=lambda x: len(x.input), sort_within_batch=True, repeat=False)

    train_dl = BatchWrapper(train_dl, ['user_id', 'country_id', 'input', 'output'], train_mask)
    dev_dl = BatchWrapper(dev_dl, ['user_id', 'country_id', 'input', 'output'], dev_mask)
    test_dl = BatchWrapper(test_dl, ['user_id', 'country_id', 'input', 'output'], test_mask)
    
    return train_dl, dev_dl, test_dl, TEXT.vocab, class_weight