# -*- coding: utf-8 -*-

import numpy as np
from numpy.core.numeric import isscalar
import copy
from pyorderedfuzzy.ofnumbers import branch
from pyorderedfuzzy.ofnumbers import defuzzy


__author__ = "amarszalek"


class OFNumber(object):
    def __init__(self, branch_f, branch_g):
        super(OFNumber, self).__init__()
        if isinstance(branch_f, branch.Branch):
            self.branch_f = branch_f
        elif isinstance(branch_f, list) or isinstance(branch_f, np.ndarray):
            self.branch_f = branch.Branch(branch_f)
        else:
            raise ValueError('branch_f must be instance of Branch, list or np.ndarray')
        if isinstance(branch_g, branch.Branch):
            self.branch_g = branch_g
        elif isinstance(branch_g, list) or isinstance(branch_g, np.ndarray):
            self.branch_g = branch.Branch(branch_g)
        else:
            raise ValueError('branch_g must be instance of Branch, list or np.ndarray')

    # deep copy operator
    def copy(self):
        return copy.deepcopy(self)

    def contains_zero(self):
        if self.branch_f.contains_zero() or self.branch_g.contains_zero():
            return True
        min_val = np.min([self.branch_f.fvalue_y, self.branch_g.fvalue_y])
        max_val = np.max([self.branch_f.fvalue_y, self.branch_g.fvalue_y])
        if min_val <= 0.0 <= max_val:
            return True
        return False

    # add method left side
    def __add__(self, right):
        res = self.copy()
        if isinstance(right, OFNumber):
            res.branch_f = res.branch_f + right.branch_f
            res.branch_g = res.branch_g + right.branch_g
        elif isscalar(right):
            res.branch_f = res.branch_f + right
            res.branch_g = res.branch_g + right
        else:
            raise ValueError('right must be instance of OFNumber class or scalar')
        return res

    # add method right side
    def __radd__(self, left):
        res = self.copy()
        if isinstance(left, OFNumber):
            res.branch_f = res.branch_f + left.branch_f
            res.branch_g = res.branch_g + left.branch_g
        elif isscalar(left):
            res.branch_f = res.branch_f + left
            res.branch_g = res.branch_g + left
        else:
            raise ValueError('left must be instance of OFNumber class or scalar')
        return res

    # sub method left side
    def __sub__(self, right):
        res = self.copy()
        if isinstance(right, OFNumber):
            res.branch_f = res.branch_f - right.branch_f
            res.branch_g = res.branch_g - right.branch_g
        elif isscalar(right):
            res.branch_f = res.branch_f - right
            res.branch_g = res.branch_g - right
        else:
            raise ValueError('right must be instance of OFNumber class or scalar')
        return res

    # sub method right side
    def __rsub__(self, left):
        res = self.copy()
        if isinstance(left, OFNumber):
            res.branch_f = left.branch_f - res.branch_f
            res.branch_g = left.branch_g - res.branch_g
        elif isscalar(left):
            res.branch_f = left - res.branch_f
            res.branch_g = left - res.branch_g
        else:
            raise ValueError('left must be instance of OFNumber class or scalar')
        return res

    # mul method left side
    def __mul__(self, right):
        res = self.copy()
        if isinstance(right, OFNumber):
            res.branch_f = res.branch_f * right.branch_f
            res.branch_g = res.branch_g * right.branch_g
        elif isscalar(right):
            res.branch_f = res.branch_f * right
            res.branch_g = res.branch_g *  right
        else:
            raise ValueError('right must be instance of OFNumber class or scalar')
        return res

    # mul method right side
    def __rmul__(self, left):
        res = self.copy()
        if isinstance(left, OFNumber):
            res.branch_f = res.branch_f * left.branch_f
            res.branch_g = res.branch_g * left.branch_g
        elif isscalar(left):
            res.branch_f = res.branch_f *  left
            res.branch_g = res.branch_g *  left
        else:
            raise ValueError('left must be instance of OFNumber class or scalar')
        return res

    # div method left side
    def __truediv__(self, right):
        res = self.copy()
        if isinstance(right, OFNumber):
            res.branch_f = res.branch_f / right.branch_f
            res.branch_g = res.branch_g / right.branch_g
        elif isscalar(right):
            res.branch_f = res.branch_f / right
            res.branch_g = res.branch_g / right
        else:
            raise ValueError('right must be instance of OFNumber class or scalar')
        return res

    # div method right side
    def __rtruediv__(self, left):
        res = self.copy()
        if isinstance(left, OFNumber):
            res.branch_f = left.branch_f / res.branch_f
            res.branch_g = left.branch_g / res.branch_g
        elif isscalar(left):
            res.branch_f = left / res.branch_f
            res.branch_g = left / res.branch_g
        else:
            raise ValueError('left must be instance of OFNumber class or scalar')
        return res

    def conjugate(self):
        res = self.copy()
        return res

    # neg method
    def __neg__(self):
        res = self.copy()
        res.branch_f = -res.branch_f
        res.branch_g = -res.branch_g
        return res

    # < method
    def __lt__(self, val):
        if isinstance(val, OFNumber):
            f = self.branch_f < val.branch_f
            g = self.branch_g < val.branch_g
        elif isscalar(val):
            f = self.branch_f < val
            g = self.branch_g < val
        else:
            raise ValueError('val must be instance of OFNumber class or scalar')
        if f and g:
            return True
        return False

    # <= method
    def __le__(self, val):
        if isinstance(val, OFNumber):
            f = self.branch_f <= val.branch_f
            g = self.branch_g <= val.branch_g
        elif isscalar(val):
            f = self.branch_f <= val
            g = self.branch_g <= val
        else:
            raise ValueError('val must be instance of OFNumber class or scalar')
        if f and g:
            return True
        return False

    # > method
    def __gt__(self, val):
        if isinstance(val, OFNumber):
            f = self.branch_f > val.branch_f
            g = self.branch_g > val.branch_g
        elif isscalar(val):
            f = self.branch_f > val
            g = self.branch_g > val
        else:
            raise ValueError('val must be instance of OFNumber class or scalar')
        if f and g:
            return True
        return False

    # >= method
    def __ge__(self, val):
        if isinstance(val, OFNumber):
            f = self.branch_f >= val.branch_f
            g = self.branch_g >= val.branch_g
        elif isscalar(val):
            f = self.branch_f >= val
            g = self.branch_g >= val
        else:
            raise ValueError('val must be instance of OFNumber class or scalar')
        if f and g:
            return True
        return False

    # == method
    def __eq__(self, val):
        if isinstance(val, OFNumber):
            f = self.branch_f == val.branch_f
            g = self.branch_g == val.branch_g
        elif isscalar(val):
            f = self.branch_f == val
            g = self.branch_g == val
        else:
            raise ValueError('val must be instance of OFNumber class or scalar')
        if f and g:
            return True
        return False

    # != method
    def __ne__(self, val):
        if isinstance(val, OFNumber):
            f = self.branch_f != val.branch_f
            g = self.branch_g != val.branch_g
        elif isscalar(val):
            f = self.branch_f != val
            g = self.branch_g != val
        else:
            raise ValueError('val must be instance of OFNumber class or scalar')
        if f or g:
            return True
        return False

    # to string method
    def __str__(self):
        return "branch_f:\n{0}\nbranch_g:\n{1}".format(self.branch_f, self.branch_g)

    def acut(self, alpha):
        fmin, fmax = self.branch_f.acut(alpha)
        gmin, gmax = self.branch_g.acut(alpha)
        return np.min([fmin, gmin]), np.max([fmax, gmax])

    def supp(self):
        return self.acut(0.0)

    def ker(self):
        return self.acut(1.0)

    # defuzzy operator
    def defuzzy(self, method='scog', args=(0.5,)):
        if method == 'scog':
            return defuzzy.scog(self, args[0])
        elif method == 'expected':
            return defuzzy.expected(self)
        else:
            raise ValueError('wrong defuzzy method')

    def order(self, dfuzzy=None, proper=False, method='scog', args=(0.5,)):
        f = self.branch_f.fvalue_y
        odr = 0
        if proper:
            g = self.branch_g.fvalue_y
            if f[0] < g[0]:
                return 1
            elif f[0] > g[0]:
                return -1
            else:
                return 0
        if dfuzzy is None:
            dfuzzy = self.defuzzy(method=method, args=args)
        if (dfuzzy is None) or (dfuzzy == f[0]):
            odr = 0
        elif dfuzzy > f[0]:
            odr = 1
        elif dfuzzy < f[0]:
            odr = -1
        return odr

    def plot_ofn(self, ax, plot_as='ordered', kwargs_f={'c': 'k'}, kwargs_g={'c': 'k'}):
        self.branch_f.plot_branch(ax, plot_as=plot_as, **kwargs_f)
        self.branch_g.plot_branch(ax, plot_as=plot_as, **kwargs_g)


# methods
# initialize OFN of linear type y = a * x + b, y = c * x + d
def init_trapezoid_abcd(a, b, c, d, dim=11):
    if dim < 2:
        raise ValueError('dim must be > 1')
    branch_f = branch.init_linear_ab(a, b, dim=dim)
    branch_g = branch.init_linear_ab(c, d, dim=dim)
    ofn = OFNumber(branch_f, branch_g)
    return ofn


# initialize OFN of linear type
# y = (fx1 - fx0) * x + fx0, y = (gx1 - gx0) * x + gx0
def init_trapezoid_x0x1(fx0, fx1, gx0, gx1, dim=11):
    if dim < 2:
        raise ValueError('dim must be > 1')
    branch_f = branch.init_linear_x0x1(fx0, fx1, dim=dim)
    branch_g = branch.init_linear_x0x1(gx0, gx1, dim=dim)
    ofn = OFNumber(branch_f, branch_g)
    return ofn


# initialize branch of gaussian type y = s * sqrt(-2 * ln(x)) + m
def init_gaussian(mf, sf, mg, sg, dim=11):
    if dim < 2:
        raise ValueError('dim must be > 1')
    branch_f = branch.init_gauss(mf, sf, dim=dim)
    branch_g = branch.init_gauss(mg, sg, dim=dim)
    ofn = OFNumber(branch_f, branch_g)
    return ofn


def init_from_scalar(scalar, dim=11):
    branch_f = branch.init_from_scalar(scalar, dim=dim)
    branch_g = branch.init_from_scalar(scalar, dim=dim)
    ofn = OFNumber(branch_f, branch_g)
    return ofn


def fabs(ofn):
    res = ofn.copy()
    res.branch_f = branch.fabs(res.branch_f)
    res.branch_g = branch.fabs(res.branch_g)
    return res


def flog(ofn):
    res = ofn.copy()
    res.branch_f = branch.flog(res.branch_f)
    res.branch_g = branch.flog(res.branch_g)
    return res


def fexp(ofn):
    res = ofn.copy()
    res.branch_f = branch.fexp(res.branch_f)
    res.branch_g = branch.fexp(res.branch_g)
    return res


def fpower(ofn, p):
    res = ofn.copy()
    res.branch_f = branch.fpower(res.branch_f, p)
    res.branch_g = branch.fpower(res.branch_g, p)
    return res


def fmax(ofn1, ofn2):
    if (not isinstance(ofn1, OFNumber)) and (not isinstance(ofn2, OFNumber)):
        raise ValueError('at least one of arguments must be a OFNumber')

    ofn_1 = init_from_scalar(ofn1, dim=ofn2.branch_f.dim) if isscalar(ofn1) else ofn1.copy()
    ofn_2 = init_from_scalar(ofn2, dim=ofn1.branch_f.dim) if isscalar(ofn2) else ofn2.copy()

    res = ofn_1.copy()
    res.branch_f = branch.fmax(ofn_1.branch_f, ofn_2.branch_f)
    res.branch_g = branch.fmax(ofn_1.branch_g, ofn_2.branch_g)
    return res


def fmin(ofn1, ofn2):
    if (not isinstance(ofn1, OFNumber)) and (not isinstance(ofn2, OFNumber)):
        raise ValueError('at least one of arguments must be a OFNumber')

    ofn_1 = init_from_scalar(ofn1, dim=ofn2.branch_f.dim) if isscalar(ofn1) else ofn1.copy()
    ofn_2 = init_from_scalar(ofn2, dim=ofn1.branch_f.dim) if isscalar(ofn2) else ofn2.copy()

    res = ofn_1.copy()
    res.branch_f = branch.fmin(ofn_1.branch_f, ofn_2.branch_f)
    res.branch_g = branch.fmin(ofn_1.branch_g, ofn_2.branch_g)
    return res
