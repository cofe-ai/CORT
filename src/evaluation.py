# coding: utf-8
#
# Copyright 2022 Hengran Zhang
# Author: Hengran Zhang
#
from sklearn.metrics import classification_report, f1_score, recall_score, precision_score,accuracy_score
from sklearn.metrics import confusion_matrix


def eva_classifier(list_t_in, list_p_in, labels=None, mask=None, average=None):
    if mask is not None:
        assert len(mask) == len(list_t_in) == len(list_p_in)
        list_t, list_p = [], []
        for t1, p1, flag in zip(list_t_in, list_p_in, mask):
            if flag > 0:
                list_t.append(t1)
                list_p.append(p1)
    else:
        list_t, list_p = list_t_in, list_p_in
    c_m = confusion_matrix(list_t, list_p, labels=labels)
    acc = accuracy_score(list_t, list_p)
    rec = recall_score(list_t, list_p, labels=labels, average=average)
    pre = precision_score(list_t, list_p, labels=labels, average=average)
    f1 = f1_score(list_t, list_p, labels=labels, average=average)
    f1_micro = f1_score(list_t, list_p, labels=labels, average='micro')
    f1_macro = f1_score(list_t, list_p, labels=labels, average='macro')
    f1_weighted = f1_score(list_t, list_p, labels=labels, average='weighted')
    pre_dist = precision_score(list_t, list_p, labels=labels, average=None)
    rec_dist = recall_score(list_t, list_p, labels=labels, average=None)
    f1_dist = f1_score(list_t, list_p, labels=labels, average=None)

    return {
        'c_m': c_m, 'acc': acc, 'f1': f1, 'pre': pre, 'rec': rec, 'f1_macro': f1_macro, 'f1_micro': f1_micro,
        'f1_weighted': f1_weighted, 'f1_dist': f1_dist, 'pre_dist': pre_dist, 'rec_dist': rec_dist
    }
