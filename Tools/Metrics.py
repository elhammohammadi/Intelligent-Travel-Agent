from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, accuracy_score
import numpy as np
from copy import deepcopy as copy
from prettytable import PrettyTable


def get_loc_session(y_true):
    
    '''* a side function *
       Given the set of target cities, return the number
       of already input cities within the same sessionions.'''
    
    loc_session = np.zeros((len(y_true)), dtype=int)
    ls = 0
    for i, y in enumerate(y_true):
        loc_session[i] = ls
        ls += 1
        if y == 0:
            ls = 0
    return loc_session


def average_metrics(true, pred):
    
    '''Given target and predicted cities, return
       accuracy, macro precision, recall and F1,
       weighted F1, and F1 on each city'''
    
    accuracy = accuracy_score(true, pred)
    macro_precision = precision_score(true, pred, average='macro')
    macro_recall = recall_score(true, pred, average='macro')
    macro_f1 = f1_score(true, pred, average='macro')
    weighted_f1 = f1_score(true, pred, average='weighted')
    label_wise = f1_score(true, pred, average=None)
    return accuracy, weighted_f1, macro_f1, macro_precision, macro_recall, label_wise
    

def calculate_sos_performance(y_true, y_pred):
    
    '''Calculate measures for the session separators
       <sos> across all searched sessions.'''
    
    t, p = copy(y_true), copy(y_pred)
    t[t != 0] = 1
    p[p != 0] = 1
    t, p = 1-t, 1-p

    accuracy = accuracy_score(t, p)
    precision = precision_score(t, p)
    recall = recall_score(t, p)
    f1 = f1_score(t, p)

    return accuracy, precision, recall, f1


def calculate_average_rank(y_true, y_prob, consider_sos=False):
    
    '''Calculate the average rank of the target cities within
       the ranked lists of potential cities.'''
    
    loc_session = get_loc_session(y_true)
    rank = np.zeros((len(y_true)))
    if consider_sos == False:
        sort_idx = np.argsort(y_prob[:, 1:], axis=1)[:, ::-1]
        rank_per_loc = [[] for _ in range(max(loc_session))]
    else:
        sort_idx = np.argsort(y_prob, axis=1)[:, ::-1]
        rank_per_loc = [[] for _ in range(max(loc_session)+1)]
    for i, yt in enumerate(y_true):
        if consider_sos == False:
            if yt == 0:
                continue
            yt -= 1
        rank[i] = np.where(sort_idx[i, :] == yt)[0][0]+1
        rank_per_loc[loc_session[i]].append(np.where(sort_idx[i, :] == yt)[0][0]+1)
    rank = np.mean(rank)
    rank_per_loc = np.array([np.mean(rpc) for rpc in rank_per_loc])
    return rank, rank_per_loc


def calculate_hit_rate(y_true, y_prob, consider_sos=False):
    
    '''Calculate Hit@k given the target cities and their
       likelihoods predicted by the model.'''
    
    loc_session = get_loc_session(y_true)
    hit_at_n = np.zeros((10))
    count = 0
    count_per_loc = np.zeros((max(loc_session)))
    if consider_sos == False:
        hit_at_n_per_loc = np.zeros((10, max(loc_session)))
        count_per_loc = np.zeros((max(loc_session)))
        sort_idx = np.argsort(y_prob[:, 1:], axis=1)[:, ::-1]
    else:
        hit_at_n_per_loc = np.zeros((10, max(loc_session)+1))
        count_per_loc = np.zeros((max(loc_session)+1))
        sort_idx = np.argsort(y_prob, axis=1)[:, ::-1]
    for i, yt in enumerate(y_true):
        if consider_sos == False:
            if yt == 0:
                continue
            yt -= 1
        for n in range(len(hit_at_n)):
            if yt in sort_idx[i, :n+1]:
                hit_at_n[n] += 1
                hit_at_n_per_loc[n, loc_session[i]] += 1
        count += 1
        count_per_loc[loc_session[i]] += 1
    hit_at_n /= count
    hit_at_n_per_loc = np.divide(hit_at_n_per_loc, count_per_loc.reshape(1, len(count_per_loc)))
    return hit_at_n, hit_at_n_per_loc


def calculate_mrr(y_true, y_prob, consider_sos=False):
    
    '''Calculate MRR@k (mean reciprocal rank) given
       the target cities and their likelihoods predicted by the model.'''
    
    loc_session = get_loc_session(y_true)
    mrr_at_n = [[] for _ in range(10)]
    if consider_sos == False:
        mrr_at_n_per_loc = [[[] for i in range(max(loc_session))] for j in range(10)]
        sort_idx = np.argsort(y_prob[:, 1:], axis=1)[:, ::-1]
    else:
        mrr_at_n_per_loc = [[[] for i in range(max(loc_session)+1)] for j in range(10)]
        sort_idx = np.argsort(y_prob, axis=1)[:, ::-1]
    for i, yt in enumerate(y_true):
        if consider_sos == False:
            if yt == 0:
                continue
            yt -= 1
        for n in range(len(mrr_at_n)):
            if yt in sort_idx[i, :n+1]:
                loc = np.where(sort_idx[i, :n+1]==yt)[0][0]+1
                mrr_at_n[n].append(1/loc)
                mrr_at_n_per_loc[n][loc_session[i]].append(1/loc)
            else:
                mrr_at_n[n].append(0)
                mrr_at_n_per_loc[n][loc_session[i]].append(0)
    mrr_at_n = np.array([np.mean(man) for man in mrr_at_n])
    mrr_at_n_per_loc = np.array([[np.mean(mpl) for mpl in man] for man in mrr_at_n_per_loc])
    return mrr_at_n, mrr_at_n_per_loc


def calculate_percentile(y_true, y_prob, consider_sos=False):
    
    '''Calculate the average percentile of the correct answer
       given the target cities and their likelihoods predicted
       by the model.'''
    
    loc_session = get_loc_session(y_true)
    percentile_at_n = [[] for _ in range(10)]
    if consider_sos == False:
        percentile_at_n_per_loc = [[[] for i in range(max(loc_session))] for j in range(10)]
        sort_idx = np.argsort(y_prob[:, 1:], axis=1)[:, ::-1]
    else:
        percentile_at_n_per_loc = [[[] for i in range(max(loc_session)+1)] for j in range(10)]
        sort_idx = np.argsort(y_prob, axis=1)[:, ::-1]
    for i, yt in enumerate(y_true):
        if consider_sos == False:
            if yt == 0:
                continue
            yt -= 1
        for n in range(len(percentile_at_n)):
            if yt in sort_idx[i, :n+1]:
                loc = np.where(sort_idx[i, :n+1]==yt)[0][0]
                percentile_at_n[n].append((n+1-loc)/(n+1))
                percentile_at_n_per_loc[n][loc_session[i]].append((n+1-loc)/(n+1))
            else:
                percentile_at_n[n].append(0)
                percentile_at_n_per_loc[n][loc_session[i]].append(0)
    percentile_at_n = np.array([np.mean(man) for man in percentile_at_n])
    percentile_at_n_per_loc = np.array([[np.mean(mpl) for mpl in man] for man in percentile_at_n_per_loc])
    return percentile_at_n, percentile_at_n_per_loc


def short_report(y_true, y_pred, y_prob):
    
    '''Generate a short performance report.'''
    
    accuracy, weighted_f1, macro_f1, _, _, _ = average_metrics(y_true, y_pred)
    rank, _ = calculate_average_rank(y_true, y_prob, consider_sos=False)
    hit_at_n, _ = calculate_hit_rate(y_true, y_prob, consider_sos=False)
    mrr_at_n, _ = calculate_mrr(y_true, y_prob, consider_sos=False)
    
    table = PrettyTable()
    table.field_names = ['Accuracy', 'Macro-F1', 'Weighted-F1', 'Average Rank', 'Hit@5', 'MRR@5']
    table.add_row(['%.2f' % el for el in [accuracy*100, macro_f1*100, weighted_f1*100, rank, hit_at_n[4]*100, mrr_at_n[4]*100]])
    
    return str(table)
    

def full_report(y_true, y_pred, y_prob, verbose=False, consider_sos=False):
    
    '''Generate the full performance report.'''
    
    accuracy, weighted_f1, macro_f1, _, _, _ = average_metrics(y_true, y_pred)
    sos_accuracy, sos_precision, sos_recall, sos_f1 = calculate_sos_performance(y_true, y_pred)
    rank, rank_per_loc = calculate_average_rank(y_true, y_prob, consider_sos=consider_sos)
    hit_at_n, hit_at_n_per_loc = calculate_hit_rate(y_true, y_prob, consider_sos=consider_sos)
    mrr_at_n, mrr_at_n_per_loc = calculate_mrr(y_true, y_prob, consider_sos=consider_sos)
    percentile_at_n, percentile_at_n_per_loc = calculate_percentile(y_true, y_prob, consider_sos=consider_sos)
    
    report_string = 'Overall Report:\n'
    table = PrettyTable()
    table.field_names = ['Accuracy', 'Macro-F1', 'Weighted-F1', 'Average Rank', 'Hit@5', 'MRR@5']
    table.add_row(['%.2f' % el for el in [accuracy*100, macro_f1*100, weighted_f1*100, rank, hit_at_n[4]*100, mrr_at_n[4]*100]])
    
    report_string += str(table) + '\n\nPerformance on <sos>:\n'
    table = PrettyTable()
    table.field_names = ['Accuracy', 'F1', 'Precision', 'Recall']
    table.add_row(['%.2f' % el for el in [sos_accuracy*100, sos_f1*100, sos_precision*100, sos_recall*100]])
    
    report_string += str(table) + '\n\nScores@k:\n'
    table = PrettyTable()
    table.field_names = ['']+['k = %d' % (k+1) for k in range(len(hit_at_n))]
    table.add_row(['Hit@k']+['%.2f' % (n*100) for n in hit_at_n])
    table.add_row(['MRR@k']+['%.2f' % (n*100) for n in mrr_at_n])
    table.add_row(['Perc@k']+['%.2f' % (n*100) for n in percentile_at_n])
    
    if verbose:
        
        report_string += str(table) + '\n\nAverage Rank per N given cities in session:\n'
        table = PrettyTable()
        table.field_names = ['N = %d' % n for n in range(len(rank_per_loc))]
        table.add_row(['%.1f' % r for r in rank_per_loc])

        report_string += str(table) + '\n\nHit@k per N given cities in session:\n'
        table = PrettyTable()
        table.field_names = ['']+['N = %d' % n for n in range(hit_at_n_per_loc.shape[1])]
        for k, row in enumerate(hit_at_n_per_loc):
            table.add_row(['k = %d' % k]+['%.1f' % (h*100) for h in row])

        report_string += str(table) + '\n\nMRR@k per N given cities in session:\n'
        table = PrettyTable()
        table.field_names = ['']+['N = %d' % n for n in range(mrr_at_n_per_loc.shape[1])]
        for k, row in enumerate(mrr_at_n_per_loc):
            table.add_row(['k = %d' % k]+['%.1f' % (m*100) for m in row])

        report_string += str(table) + '\n\nPerc@k per N given cities in session:\n'
        table = PrettyTable()
        table.field_names = ['']+['N = %d' % n for n in range(percentile_at_n_per_loc.shape[1])]
        for k, row in enumerate(percentile_at_n_per_loc):
            table.add_row(['k = %d' % k]+['%.1f' % (p*100) for p in row])
        
    report_string += str(table)
    
    return report_string
    