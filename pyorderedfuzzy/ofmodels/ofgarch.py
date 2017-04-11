# -*- coding: utf-8 -*-

import numpy as np
from pyorderedfuzzy.ofnumbers.ofnumber import OFNumber
from pyorderedfuzzy.ofmodels.ofseries import OFSeries
from scipy.optimize import minimize
from pyorderedfuzzy.ofmodels.oflinreg import ofns2array, array2ofns
#from pyorderedfuzzy.ofmodels._objective import obj_func_ar_ls, obj_func_ar_cml

__author__ = "amarszalek"


class OFGARCH(object):
    def __init__(self, order_p=1, order_q=1, intercept=True):
        super(OFGARCH, self).__init__()
        self.intercept = intercept
        self.order_p = order_p
        self.order_q = order_q
        self.coef = OFSeries([])
        self.residuals = OFSeries([])

    def fit(self, ofseries, order_p, order_q, intercept=True, solver='L-BFGS-B', options={}):
        dim = ofseries[0].branch_f.dim
        self.order_p = order_p
        self.order_q = order_q
        self.intercept = intercept
        n_coef = order_p + order_q + 1
        if self.intercept:
            n_coef += 1

        # initial coef
        ran = np.random.random(n_coef)
        coef = [OFNumber(np.ones(dim)*r, np.ones(dim)*r) for r in ran]

        if solver == 'L-BFGS-B':
            if options == {}:
                options = {'disp': None, 'gtol': 1.0e-12, 'eps': 1e-08, 'maxiter': 1000, 'ftol': 2.22e-09}
            p0 = ofns2array(coef)
            ofns = ofns2array(ofseries)
            args = (order_p, order_q, n_coef, dim, ofns, intercept)
            res = minimize(fun_obj_qml, p0, args=args, method='L-BFGS-B', jac=True, options=options)
            coef = array2ofns(res.x, n_coef, dim)
            self.coef = OFSeries(coef)
        elif solver == 'CL-BFGS-B':
            if options == {}:
                options = {'disp': None, 'gtol': 1.0e-12, 'eps': 1e-08, 'maxiter': 1000, 'ftol': 2.22e-09}
            p0 = ofns2array(coef)
            ofns = ofns2array(ofseries)
            args = (n_coef, dim, ofns, self.intercept)
            res = minimize(fun_obj_qml_c, p0, args=args, method='L-BFGS-B', jac=True, options=options)
            coef = array2ofns(res.x, n_coef, dim)
            self.coef = OFSeries(coef)
        else:
            raise ValueError('wrong solver')


def compute_ht(coef, past_x, past_h):
    ht = coef[0]
    for p in range(1, len(past_x)+1):
        ht = ht + coef[p] * past_x[-p]
    p = len(past_x) + 1
    for q in range(1, len(past_h)+1):
        ht = ht + coef[p+q] * past_x[-q]
    return ht


def fun_obj_qml(p, order_p, order_q, n_coef, dim, ofns, intercept):
    e = 0.0
    n_cans = int(len(ofns) / (2 * dim))
    can = ofns.reshape((n_cans, 2 * dim))
    coef = p.reshape((n_coef, 2 * dim))
    grad = np.zeros(len(p))
    if intercept:
        past_x = np.zeros((order_p, 2 * dim))
        beta = 1.0
        for i in range(order_q):
            beta = beta - coef[order_p+2+i]
        past_h = np.array([coef[1]/beta for _ in range(order_q)])
        for i in range(n_cans):
            r = can[i] - coef[0]
            ht = compute_ht(coef[1:], past_x, past_h)
            past_x[:-1] = past_x[1:]
            past_x[-1] = r*r
            past_h[:-1] = past_h[1:]
            past_h[-1] = ht
            e += np.sum((0.5*r*r)/ht + np.log(ht))
            grad[:2 * dim] -= r/ht
            grad[2*dim:4*dim] += 0.5*(1/ht - (r*r)/(ht*ht))
            for j in range(2, order_p+2):
                grad[2 * dim * j:2 * dim * (j + 1)] += 0.5*(1/ht - (r*r)/(ht*ht))*past_x[-2]*past_x[-2]
            for j in range(order_p+2, order_p+order_q+2):
                grad[2 * dim * j:2 * dim * (j + 1)] +=  0.5*(1/ht - (r*r)/(ht*ht))*past_h[-2]*past_h[-2]
    else:
        past_x = np.zeros((order_p, 2 * dim))
        beta = 1.0
        for i in range(order_q):
            beta = beta - coef[order_p + 1 + i]
        past_h = np.array([coef[0] / beta for _ in range(order_q)])
        for i in range(n_coef, n_cans):
            r = can[i]
            ht = compute_ht(coef, past_x, past_h)
            past_x[:-1] = past_x[1:]
            past_x[-1] = can[i]
            past_h[:-1] = past_h[1:]
            past_h[-1] = ht
            e += np.sum((0.5 * r * r) / ht + np.log(ht))
            #for j in range(n_coef):
            #    grad[2 * dim * j:2 * dim * (j + 1)] -= 2 * r * can[i - j - 1]
    return e, grad


def fun_obj_qml_c():
    pass