"""
This file contains functions for evaluating macro-f1 and accuracy
"""

from sklearn.metrics import f1_score
from collections import OrderedDict


def transform_to_index(trues, preds):

    true_map = dict([(long_term, index) for index, long_term in enumerate(list(OrderedDict.fromkeys(trues)))])
    tag_cnt = len(true_map)
    true_index = [true_map[t] for t in trues]

    for pred in preds:
        if pred not in true_map:
            true_map[pred] = len(true_map)
    pred_index = [true_map[pred] for pred in preds]
    return true_index, pred_index, tag_cnt


def macro_f1(trues, preds):
    true_index, pred_index, tag_cnt = transform_to_index(trues, preds)
    macro_f1 = f1_score(true_index, pred_index, average=None)[:tag_cnt]
    macro_f1 = sum(macro_f1) / len(macro_f1)
    return macro_f1


def accuracy(trues, preds):
    acc_cnt = 0
    for index, true in enumerate(trues):
        pred = preds[index]
        if pred == true:
            acc_cnt += 1
    acc = acc_cnt * 1.0 / len(trues)
    return acc

