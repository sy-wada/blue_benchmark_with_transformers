# coding=utf-8
"""
Original code is derived from BLUE_Benchmark
https://github.com/ncbi-nlp/BLUE_Benchmark/blob/master/blue/eval_hoc.py

We modified some problems:
 - LABELS: 'data' key error occurs in original code.
 - counts only positive labels:
 - I/O for our implimentation.
"""

import pandas as pd
import numpy as np

from sklearn.metrics import roc_curve, auc

# Compute ROC curve and ROC area for each class
def eval_roc_auc(y_true, pred_score, num_labels):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(num_labels):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], pred_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), pred_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    return roc_auc

#ADD:
LABELS = []
for i in range(10):
        LABELS.append('{}_1'.format(i))

def get_p_r_f_arrary(test_predict_label, test_true_label):
    num, cat = test_predict_label.shape
    acc_list = []
    prc_list = []
    rec_list = []
    f_score_list = []
    for i in range(num):
        label_pred_set = set()
        label_gold_set = set()

        for j in range(cat):
            if test_predict_label[i, j] == 1:
                label_pred_set.add(j)
            if test_true_label[i, j] == 1:
                label_gold_set.add(j)

        uni_set = label_gold_set.union(label_pred_set)
        intersec_set = label_gold_set.intersection(label_pred_set)

        tt = len(intersec_set)
        if len(label_pred_set) == 0:
            prc = 0
        else:
            prc = tt / len(label_pred_set)

        acc = tt / len(uni_set)

        rec = tt / len(label_gold_set)

        if prc == 0 and rec == 0:
            f_score = 0
        else:
            f_score = 2 * prc * rec / (prc + rec)

        acc_list.append(acc)
        prc_list.append(prc)
        rec_list.append(rec)
        f_score_list.append(f_score)

    mean_prc = np.mean(prc_list)
    mean_rec = np.mean(rec_list)
    f_score = 2 * mean_prc * mean_rec / (mean_prc + mean_rec)
    return mean_prc, mean_rec, f_score


def eval_hoc(df, mode):
    """
    df is DataFrame of pandas. It needs 3 columns below:
    index, labels(=y_true), pred_labels(=y_preds)
    We create two DataFrames (true_df, pred_df) from df and rename 'pred_labels' as 'labels' in pred_df.
    """
    data = {}

    true_df = df.drop('pred_labels', axis=1)
    pred_df = df.drop('labels', axis=1).rename(columns={'pred_labels': 'labels'})

    assert len(true_df) == len(pred_df), \
        f'Gold line no {len(true_df)} vs Prediction line no {len(pred_df)}'

    for i in range(len(true_df)):
        true_row = true_df.iloc[i]
        pred_row = pred_df.iloc[i]
        assert true_row['index'] == pred_row['index'], \
            'Index does not match @{}: {} vs {}'.format(i, true_row['index'], pred_row['index'])

        key = true_row['index'][:true_row['index'].find('_')]
        if key not in data:
            data[key] = (set(), set())

        if not pd.isna(true_row['labels']):
            for l in true_row['labels'].split(','):
                if l.endswith('_1'):      #ADD:
                    data[key][0].add(LABELS.index(l))

        if not pd.isna(pred_row['labels']):
            for l in pred_row['labels'].split(','):
                if l.endswith('_1'):      #ADD:
                    data[key][1].add(LABELS.index(l))
    
    #ADD:
    if mode == 'test':
        num_docs = 315
    elif mode == 'dev':
        num_docs = 157

    assert len(data) == num_docs, \
        'There are {} documents in the {} set: %d'.format(num_docs, mode) % len(data)

    y_test = []
    y_pred = []
    for k, (true, pred) in data.items():
        t = [0] * len(LABELS)
        for i in true:
            t[i] = 1

        p = [0] * len(LABELS)
        for i in pred:
            p[i] = 1

        y_test.append(t)
        y_pred.append(p)

    y_test = np.array(y_test)
    y_pred = np.array(y_pred)

    p, r, f1 = get_p_r_f_arrary(y_pred, y_test)
    results = {
        "precision": p,
        "recall": r,
        "FB1": f1,
    }
    return results
