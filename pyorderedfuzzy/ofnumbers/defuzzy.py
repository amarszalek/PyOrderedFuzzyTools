# -*- coding: utf-8 -*-

import numpy as np

__author__ = "amarszalek"


# center of gravity operator
def scog(ofn, alpha):
    f = ofn.branch_f.fvalue_y
    g = ofn.branch_g.fvalue_y

    if (f == g).all():
        if f.max() == f.min():
            return f[0]
        else:
            return None

    mian = (np.abs(g - f)).sum()
    mian -= 0.5 * np.abs(g[0] - f[0])
    mian -= 0.5 * np.abs(g[-1] - f[-1])
    licz = (((alpha * f + (1.0 - alpha) * g)) * np.abs(g - f)).sum()
    licz -= 0.5 * ((alpha * f[0] + (1.0 - alpha) * g[0])) * np.abs(g[0] - f[0])
    licz -= 0.5 * ((alpha * f[-1] + (1.0 - alpha) * g[-1])) * np.abs(g[-1] - f[-1])
    return licz / mian


# expected value operator
def expected(self):
    f = self.branch_f.fvalue_y
    g = self.branch_g.fvalue_y
    dx = 1.0 / (self.branch_f.dim - 1)
    cal = (f + g).sum()
    cal -= 0.5 * (f[0] + g[0])
    cal -= 0.5 * (f[-1] + g[-1])
    return 0.5 * cal * dx
