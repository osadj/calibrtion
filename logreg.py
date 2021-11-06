#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 19:04:02 2021

@author: Omid Sadjadi <omid.sadjadi@ieee.org>
"""

import numpy as np
from scipy.special import expit, xlogy
from scipy.optimize import fmin_l_bfgs_b


def platt_calibration(f, y):
    """Classifier output calibration using logistic regression (aka Platt Scaling)
    Parameters
    ----------
    f : ndarray of shape (n_samples, n_features)
        Classifier/detecor output
    y : ndarray of shape (n_samples,)
        Targets in {0, 1} or {-1, 1}

    Returns
    -------
    A : float
        The regression slope
    B : float
        The regression intercept
        
    References
    ----------
    J. Platt, "Probabilistic outputs for support vector machines and comparisons to 
              regularized likelihood methods," Advances in Large Margin Classifiers,
              10(3), pp.61-74, 1999.
    """

    f = np.concatenate((f, np.ones((f.shape[0], 1))), axis=1)

    # Setting Bayesian priors
    prior0 = sum(y <= 0)
    prior1 = y.size - prior0
    pi = prior1 / y.size
    T = np.zeros(y.size)
    T[y > 0] = (prior1 + 1) / (prior1 + 2)
    T[y <= 0] = 1 / (prior0 + 2)
    a = np.log(pi /(1 - pi)) # log prior odds

    def objective(AB):
        P = expit(f @ AB + a)
        logloss = -(xlogy(T, P) + xlogy(1 - T, 1 - P))
        return logloss.sum()

    def gradient(AB):
        P = expit(f @ AB + a)
        error = P - T
        dAB = f.T @ error
        return dAB

    AB0 = np.array([0.0, np.log((prior0 + 1.0) / (prior1 + 1.0))])
    A, B = fmin_l_bfgs_b(objective, AB0, fprime=gradient, disp=False)[0]
    return A, B
