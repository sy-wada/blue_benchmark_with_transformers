# coding=utf-8

from scipy.stats import pearsonr

def eval_sts(x, y):
    r, p = pearsonr(x, y)
    return r