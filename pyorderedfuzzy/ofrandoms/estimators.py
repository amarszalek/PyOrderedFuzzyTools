# -*- coding: utf-8 -*-

import numpy as np
from pyorderedfuzzy.ofnumbers.ofnumber import OFNumber, fmax

__author__ = "amarszalek"


def sample_mu(ofseries, ord_method='expected'):
    n = len(ofseries)
    suma = ofseries[0] if ofseries[0].order(method=ord_method) >= 0 else OFNumber(ofseries[0].branch_g,
                                                                                  ofseries[0].branch_f)
    for i in range(1, n):
        if ofseries[i].order(method=ord_method) >= 0:
            suma = suma + ofseries[i]
        else:
            suma = suma + OFNumber(ofseries[i].branch_g, ofseries[i].branch_f)
    return suma / n


def sample_sig2(ofseries, ord_method='expected', ddof=1):
    n = len(ofseries)
    mu = sample_mu(ofseries)
    s2 = sample_s2(ofseries, ddof=ddof)
    suma = (ofseries[0] - mu) * (ofseries[0] - mu) if ofseries[0].order(method=ord_method) >= 0 else \
        (OFNumber(ofseries[0].branch_g, ofseries[0].branch_f)-mu) * (OFNumber(ofseries[0].branch_g,
                                                                              ofseries[0].branch_f)-mu)
    for i in range(1, n):
        if ofseries[i].order(method=ord_method) >= 0:
            suma = suma + (ofseries[i] - mu) * (ofseries[i] - mu)
        else:
            suma = suma + (OFNumber(ofseries[i].branch_g, ofseries[i].branch_f)-mu) * \
                          (OFNumber(ofseries[i].branch_g, ofseries[i].branch_f)-mu)
    suma = suma/(n-ddof)
    sig2 = suma - s2
    return fmax(sig2, 0.0)


def sample_s2(ofseries, ddof=1):
    defuzz = np.array([ofn.defuzzy(method='expected') for ofn in ofseries])
    return np.var(defuzz, ddof=ddof)


def sample_p(ofseries, ord_method='expected'):
    n = len(ofseries)
    p = 0.0
    for el in ofseries:
        ord_el = el.order(method=ord_method)
        if ord_el >= 0:
            p += 1.0
    return p/n


def sample_mean(ofseries):
    arr = np.array(ofseries, dtype=object)
    return np.mean(arr)


def sample_var(ofseries, ddof=1):
    arr = np.array(ofseries, dtype=object)
    return np.var(ofseries, ddof=ddof)
