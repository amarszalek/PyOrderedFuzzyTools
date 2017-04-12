# -*- coding: utf-8 -*-

import numpy as np
from pyorderedfuzzy.ofnumbers.ofnumber import OFNumber
from pyorderedfuzzy.ofmodels.ofseries import OFSeries
from arch import arch_model

__author__ = "amarszalek"


class OFGARCH(object):
    def __init__(self):
        super(OFGARCH, self).__init__()
        self.order_p = 0
        self.order_q = 0
        self.coef = OFSeries([])
        self.tmean = 'zero'
        self.residuals = OFSeries([])

    def fit(self, ofns, order_p, order_q, tmean='zero'):
        dim = ofns[0].branch_f.dim
        n = len(ofns)
        self.order_p = order_p
        self.order_q = order_q
        n_coef = order_p + order_q + 1
        if tmean == 'constant':
            n_coef += 1

        self.coef = OFSeries([OFNumber(np.zeros(dim), np.zeros(dim)) for _ in range(n_coef)])

        # branch f
        for i in range(dim):
            data = np.array([ofns[t].branch_f.fvalue_y[i] for t in range(n)])
            garch = arch_model(data, mean=tmean, p=order_p, q=order_q)
            res = garch.fit(disp='off')
            for c in range(n_coef):
                self.coef[c].branch_f.fvalue_y[i] = res.params[c]
        # branch g
        for i in range(dim):
            data = np.array([ofns[t].branch_g.fvalue_y[i] for t in range(n)])
            garch = arch_model(data, mean=tmean, p=order_p, q=order_q)
            res = garch.fit(disp='off')
            for c in range(n_coef):
                self.coef[c].branch_g.fvalue_y[i] = res.params[c]
