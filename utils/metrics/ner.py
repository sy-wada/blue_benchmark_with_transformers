# coding=utf-8
"""
The Original code (https://github.com/ncbi-nlp/bluebert/blob/master/bluebert/conlleval.py) has some troubles:
 - starting with I
 - I next to O
Then, we solve them in a primitive way.
 - we make all tags into a one-dimensional array and ignore blank lines (=break point).
 - All phrases are assumed to start with "B" so that disjoint mentions can be combined.
"""

import os

def get_phrase(phraselist, tag, index):
    if tag == 'B':
        phraselist.append([str(index)])
    elif tag == 'I':
        phraselist[-1].append(str(index))
    return

def eval_ner(iterable):
    y_true = []
    y_pred = []
    num_token = 0
    for i, line in enumerate(iterable):
        try:
            token, true, pred = line.strip().split(' ')
        except:
            continue
        get_phrase(y_true, true, i)
        get_phrase(y_pred, pred, i)
        num_token += 1
        
    y_true = set(map('_'.join, y_true))
    y_pred = set(map('_'.join, y_pred))

    TP = len(y_true & y_pred)
    FN = len(y_true) - TP
    FP = len(y_pred) - TP
    prec = TP / (TP + FP)
    rec = TP / (TP + FN)
    fb1 = 2 * rec * prec / (rec + prec)
    
    results = {
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "precision": prec,
        "recall": rec,
        "FB1": fb1,
    }
    report = 'processed {} tokens with {} phrases; found: {} phrases; correct: {}.\n'.format(num_token, len(y_true), len(y_pred), TP)
    report += 'TP: {}, FP: {}, FN: {}\n'.format(TP, FP, FN)
    report += 'Precision: {:.2f}%, Recall: {:.2f}%, FB1: {:.2f}%'.format(
                                                                prec * 100,
                                                                rec * 100,
                                                                fb1 * 100)
    return results, report
