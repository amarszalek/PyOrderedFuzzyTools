# -*- coding: utf-8 -*-

import numpy as np
from pyorderedfuzzy.ofnumbers.ofnumber import OFNumber
from pyorderedfuzzy.ofmodels.ofseries import OFSeries
from scipy.optimize import minimize
from pyorderedfuzzy.ofmodels.oflinreg import ofns2array, array2ofns
from pyorderedfuzzy.ofmodels._objective import obj_func_ar_ls

__author__ = "amarszalek"


class OFAutoRegressive(object):
    def __init__(self, order=1, intercept=True, coef=[], initial=[]):
        super(OFAutoRegressive, self).__init__()
        self.intercept = intercept
        self.order = order
        self.coef = OFSeries(coef)
        self.initial = initial

    def fit(self, ofseries, order, intercept=True, method='ls', solver='L-BFGS-B', options={}):
        dim = ofseries[0].branch_f.dim
        self.order = order
        self.intercept = intercept
        self.initial = ofseries[-order-1:]
        n_coef = order
        if self.intercept:
            n_coef += 1

        # initial coef
        ran = np.random.random(n_coef)
        coef = [OFNumber(np.ones(dim)*r, np.ones(dim)*r) for r in ran]

        if solver == 'LU':
            pass
        elif solver == 'L-BFGS-B':
            if options == {}:
                options = {'disp': None, 'gtol': 1.0e-12, 'eps': 1e-08, 'maxiter': 1000, 'ftol': 2.22e-09}
            p0 = ofns2array(coef)
            ofns = ofns2array(ofseries)
            args = (order, n_coef, dim, ofns, intercept)
            if method == 'ls':
                res = minimize(fun_obj_ols, p0, args=args, method='L-BFGS-B', jac=True, options=options)
            elif method == 'ml':
                res = minimize(fun_obj_mle, p0, args=args, method='L-BFGS-B', jac=True, options=options)
            else:
                raise ValueError('wrong method')
            coef = array2ofns(res.x, n_coef, dim)
            self.coef = OFSeries(coef)
        elif solver == 'C_L-BFGS-B':
            if options == {}:
                options = {'disp': None, 'gtol': 1.0e-12, 'eps': 1e-08, 'maxiter': 1000, 'ftol': 2.22e-09}
            p0 = ofns2array(coef)
            ofns = ofns2array(ofseries)
            args = (n_coef, dim, ofns, self.intercept)
            if method == 'ls':
                res = minimize(fun_obj_ols_c, p0, args=args, method='L-BFGS-B', jac=True, options=options)
            elif method == 'ml':
                res = minimize(fun_obj_mle_c, p0, args=args, method='L-BFGS-B', jac=True, options=options)
            else:
                raise ValueError('wrong method')
            coef = array2ofns(res.x, n_coef, dim)
            self.coef = OFSeries(coef)
        else:
            raise ValueError('wrong solver')

        # residuals = []
        # for i in range(order, len(candles)):
        #     if i < order:
        #         residuals.append(OFNumber(np.zeros(dim), np.zeros(dim)))
        #     else:
        #         pred = self.predict(1, initial=candles[i-order:i])
        #         residuals.append(candles[i]-pred[0])
        # self.residuals = residuals

    def predict(self, n, initial=None, mean=None):
        if initial is None:
            initial = self.initial
        predicted = []
        for t in range(1, n+1):
            if self.intercept:
                y = self.coef[0]
                for p in range(1, self.order+1):
                    y = y + self.coef[p] * initial[-p]
            else:
                y = self.coef[0] * initial[-1]
                for p in range(1, self.order):
                    y = y + self.coef[p] * initial[-p-1]
            if mean is not None:
                y = y + mean
            predicted.append(y)
            initial.append(y)
        return predicted


def autoreg(coef, past, intercept):
    if intercept:
        ar = np.dot(coef[1:], past[::-1])
        ar = ar + coef[0]
    else:
        ar = np.dot(coef, past[::-1])
    return ar


def fun_obj_ols(p, order, n_coef, dim, ofns, intercept):
    e = 0.0
    n_cans = int(len(ofns)/(2 * dim))
    can = ofns.reshape((n_cans, 2 * dim))
    coef = p.reshape((n_coef, 2 * dim))
    grad = np.zeros(len(p))
    if intercept:
        for i in range(order, n_cans):
            r = can[i] - autoreg(coef, can[i-order:i], intercept)
            e += np.sum(r * r)
            grad[:2 * dim] -= 2.0 * r
            for j in range(1, n_coef):
                grad[2 * dim * j:2 * dim * (j + 1)] -= 2 * r * can[i - j]
    else:
        for i in range(n_coef, n_cans):
            r = can[i] - autoreg(coef, can[i-n_coef:i], intercept)
            e += np.sum(r * r)
            for j in range(n_coef):
                grad[2 * dim * j:2 * dim * (j + 1)] -= 2 * r * can[i - j-1]
    return e, grad


def fun_obj_ols_c(p, n_coef, dim, ofns, intercept):
    if intercept:
        res = obj_func_ar_ls(p, ofns, n_coef, dim * 2, 1)
    else:
        res = obj_func_ar_ls(p, ofns, n_coef, dim * 2, 0)
    return res[0], np.array(res[1])


def fun_obj_mle():
    pass


def fun_obj_mle_c():
    pass