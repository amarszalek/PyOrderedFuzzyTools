# -*- coding: utf-8 -*-

import numpy as np
from pyorderedfuzzy.ofnumbers.ofnumber import OFNumber
from pyorderedfuzzy.ofmodels.ofseries import OFSeries, init_from_scalar_values

__author__ = "amarszalek"


def ofnormal(mu, sig2, s2, p):
    dim = mu.branch_f.dim
    minf = np.min(mu.branch_f.fvalue_y)
    ming = np.min(mu.branch_g.fvalue_y)
    c = 0.0
    if min(minf, ming) <= 0.0:
        c = np.abs(min(minf, ming)) + 1.0
    eta = OFNumber(np.ones(dim)*c, np.ones(dim)*c) + mu
    x = np.zeros(dim)
    y = np.zeros(dim)
    for i in range(dim):
        if sig2.branch_f.fvalue_y[i] <= 0.0:
            x[i] = 1.0
        else:
            x[i] = np.random.normal(1, np.sqrt(sig2.branch_f.fvalue_y[i]) / eta.branch_f.fvalue_y[i])
        if sig2.branch_g.fvalue_y[i] <= 0.0:
            y[i] = 1.0
        else:
            y[i] = np.random.normal(1, np.sqrt(sig2.branch_g.fvalue_y[i]) / eta.branch_g.fvalue_y[i])
    r = np.random.random()
    s = np.random.normal(0, np.sqrt(s2))
    if r < p:
        ksi = OFNumber(x, y)*eta+s
    else:
        ksi = OFNumber(x, y)*OFNumber(eta.branch_g, eta.branch_f)+s
    return ksi-c


def ofnormal_sample(n, mu, sig2, s2, p):
    sample = map(lambda i: ofnormal(mu, sig2, s2, p), range(n))
    return OFSeries(list(sample))


def ofnormal_scalar(mu, sig, dim):
    r = np.random.normal(mu, sig)
    return OFNumber(np.ones(dim)*r, np.ones(dim)*r)


def ofnormal_scalar_sample(n, mu, sig, dim):
    r = np.random.normal(mu, sig, size=(n,))
    return init_from_scalar_values(r, dim)
