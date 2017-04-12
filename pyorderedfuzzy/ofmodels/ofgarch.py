# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
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

    def fit(self, pd_ofns, order_p, order_q, tmean='zero', last_obs=None):
        ofns = np.array(pd_ofns)
        self.index = pd_ofns.index
        self.dim = ofns[0].branch_f.dim
        n = len(ofns)
        self.order_p = order_p
        self.order_q = order_q
        n_coef = order_p + order_q + 1
        if tmean == 'constant':
            n_coef += 1

        self.coef = OFSeries([OFNumber(np.zeros(self.dim), np.zeros(self.dim)) for _ in range(n_coef)])

        # branch f
        self.models_f = []
        for i in range(self.dim):
            data = np.array([ofns[t].branch_f.fvalue_y[i] for t in range(n)])
            data = pd.Series(data, index=self.index)
            garch = arch_model(data, mean=tmean, p=order_p, q=order_q)
            res = garch.fit(disp='off', last_obs=last_obs)
            self.models_f.append(res)
            for c in range(n_coef):
                self.coef[c].branch_f.fvalue_y[i] = res.params[c]
        # branch g
        self.models_g = []
        for i in range(self.dim):
            data = np.array([ofns[t].branch_g.fvalue_y[i] for t in range(n)])
            data = pd.Series(data, index=self.index)
            garch = arch_model(data, mean=tmean, p=order_p, q=order_q)
            res = garch.fit(disp='off', last_obs=last_obs)
            self.models_g.append(res)
            for c in range(n_coef):
                self.coef[c].branch_g.fvalue_y[i] = res.params[c]

    def forecast(self, horizon=1, start=None, align='target'):
        predicted = np.array([[OFNumber(np.zeros(self.dim), np.zeros(self.dim)) for _ in range(horizon)]
                              for _ in range(len(self.index))], dtype=object)
        for i in range(self.dim):
            forecasts = self.models_f[i].forecast(horizon=horizon, align=align, start=start)
            var = np.array(forecasts.variance)
            for t in range(len(self.index)):
                for h in range(horizon):
                    predicted[t, h].branch_f.fvalue_y[i] = var[t, h]
        for i in range(self.dim):
            forecasts = self.models_g[i].forecast(horizon=horizon, align=align, start=start)
            var = np.array(forecasts.variance)
            for t in range(len(self.index)):
                for h in range(horizon):
                    predicted[t, h].branch_g.fvalue_y[i] = var[t, h]
        return pd.DataFrame(predicted, index=self.index)
