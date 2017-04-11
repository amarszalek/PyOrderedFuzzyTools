# -*- coding: utf-8 -*-

import numpy as np
from pyorderedfuzzy.ofnumbers.ofnumber import OFNumber
from pyorderedfuzzy.ofmodels.ofseries import OFSeries
from scipy.optimize import minimize
from pyorderedfuzzy.ofmodels._objective import obj_func_lin_reg

__author__ = "amarszalek"


class OFLinearRegression(object):
    def __init__(self, intercept=True):
        super(OFLinearRegression, self).__init__()
        self.intercept = intercept
        self.n_coef = 0
        self.coef = OFSeries([])
        self.residuals = OFSeries([])

    # x, y -> np.array of OFN
    # x.shape = (T, N), y.shape = (T,)
    def fit(self, x, y, solver='L-BFGS-B', options={}):
        dim = x[0, 0].branch_f.dim
        self.n_coef = x.shape[1]
        if self.intercept:
            self.n_coef += 1
            x = add_one(x)

        # initial coef
        ran = np.random.random(self.n_coef)
        coef = np.array([OFNumber(np.ones(dim)*r, np.ones(dim)*r) for r in ran], dtype=object)

        if solver == 'LU':
            pass
        elif solver == 'L-BFGS-B':
            if options == {}:
                options = {'disp': None, 'gtol': 1e-08, 'eps': 1e-08, 'maxiter': 1000, 'ftol': 2.22e-09}
            p0 = ofns2array(coef)
            args = (self.n_coef, dim, x, y)
            res = minimize(fun_obj_ols, p0, args=args, method='L-BFGS-B', jac=True, options=options)
            coef = array2ofns(res.x, self.n_coef, dim)
            self.coef = OFSeries(coef)
        elif solver == 'CL-BFGS-B':
            if options == {}:
                options = {'disp': None, 'gtol': 1e-08, 'eps': 1e-08, 'maxiter': 1000, 'ftol': 2.22e-09}
            p0 = ofns2array(coef)
            xx = ofns2array(x.reshape(x.shape[0]*x.shape[1]))
            yy = ofns2array(y)
            args = (self.n_coef, dim, xx, yy)
            res = minimize(fun_obj_ols_c, p0, args=args, method='L-BFGS-B', jac=True, options=options)
            coef = array2ofns(res.x, self.n_coef, dim)
            self.coef = OFSeries(coef)
        else:
            raise ValueError('wrong solver')

        y_prog = np.apply_along_axis(lambda v: linreg(self.coef, v), 1, x)
        self.residuals = OFSeries(y-y_prog)

    def predict(self, x):
        if self.intercept:
            x = add_one(x)
        y_prog = np.apply_along_axis(lambda v: linreg(self.coef, v), 1, x)
        return OFSeries(y_prog)


def add_one(data):
    dim = data[0, 0].branch_f.dim
    one = OFNumber(np.ones(dim), np.ones(dim))
    data = data.tolist()
    for i in range(len(data)):
        data[i].insert(0, one)
    return np.array(data, dtype=object)


def ofns2array(coef):
    p = coef[0].to_array()
    for i in range(1, len(coef)):
        p = np.concatenate((p, coef[i].to_array()))
    return np.array(p)


def array2ofns(arr, n, dim):
    ofns = []
    for nc in range(n):
        s1 = nc*2*dim
        s2 = nc*2*dim + dim
        ofns.append(OFNumber(arr[s1:s2], arr[s2:s2+dim]))
    return np.array(ofns, dtype=object)


def linreg(coef, x):
    return np.dot(np.array(coef, dtype=object), x)


def fun_obj_ols(p, n_coef, dim, x, y):
    coef = array2ofns(p, n_coef, dim)
    y_prog = np.apply_along_axis(lambda v: linreg(coef, v), 1, x)
    r = (y - y_prog)
    e = map(lambda v: v.to_array(), r*r)
    e = np.sum(list(e))
    grad = []
    for i in range(n_coef):
        g = np.sum(-2*r*x[:, i])
        grad.extend(g.to_array())
    return e, np.array(grad)


def fun_obj_ols_c(p, n_coef, dim, xx, yy):
    res = obj_func_lin_reg(p, xx, yy, n_coef, dim*2)
    return res[0], np.array(res[1])
