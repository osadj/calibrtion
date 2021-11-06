#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 18:51:18 2021

@author: Omid Sadjadi <omid.sadjadi@ieee.org>
"""

import numpy as np
from sklearn.metrics import det_curve
from logreg import platt_calibration


def main():
    # generate toy target and non-target scores
    np.random.seed(42)
    n_tar = 1000
    n_non = 10000
    tar = np.random.randn(n_tar)
    non = np.random.randn(n_non) - 3
    scores = np.r_[tar, non]
    labels = np.zeros(scores.size)
    labels[:n_tar] = 1

    # compute calibration parameters using Platt Scaling (logistic regression)
    a, b = platt_calibration(scores.reshape(-1, 1), labels)
    scores_calibrated = a * scores + b

    # compute minimum and actual DCF for raw scores
    p_target = 0.01
    beta = (1 - p_target)/p_target
    theta = np.log(beta)

    fpr, fnr = det_curve(labels, scores)[:2]
    min_dcf = min(fnr + beta * fpr)

    miss = 1 - sum(labels[scores >  theta]) / sum(labels)
    fa = sum(1 - labels[scores >  theta]) / sum(1 - labels)
    act_dcf = miss + beta * fa

    print(f'Before: minDCF={min_dcf:.3f}, actDCF={act_dcf:.3f}')

    # compute minimum and actual DCF for calibrated scores
    fpr, fnr = det_curve(labels, scores_calibrated)[:2]
    min_dcf = min(fnr + beta * fpr)

    miss = 1 - sum(labels[scores_calibrated >  theta]) / sum(labels)
    fa = sum(1 - labels[scores_calibrated >  theta]) / sum(1 - labels)
    act_dcf = miss + beta * fa

    print(f'After: minDCF={min_dcf:.3f}, actDCF={act_dcf:.3f}')


if __name__ == '__main__':

    main()
