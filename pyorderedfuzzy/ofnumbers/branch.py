# -*- coding: utf-8 -*-

import numpy as np
from numpy.core.numeric import isscalar
import copy

__author__ = "amarszalek"


class Branch(object):
    def __init__(self, fvalue_y):
        super(Branch, self).__init__()
        if len(fvalue_y) == 0:
            raise ValueError('fvalue_y is empty')
        self.dim = len(fvalue_y)
        self.domain_x = np.linspace(0., 1., self.dim)
        self.fvalue_y = np.asarray(fvalue_y, dtype=np.double)

    # deep copy operator
    def copy(self):
        return copy.deepcopy(self)

    def contains_zero(self):
        min_val = np.min(self.fvalue_y)
        max_val = np.max(self.fvalue_y)
        if min_val <= 0.0 <= max_val:
            return True
        return False

    # add method left side
    def __add__(self, right):
        res = self.copy()
        if isinstance(right, Branch):
            if res.dim != right.dim:
                raise ValueError('self.dim and right.dim must be equal')
            res.fvalue_y = res.fvalue_y + right.fvalue_y
        elif isscalar(right):
            res.fvalue_y = res.fvalue_y + right
        else:
            raise ValueError('right must be instance of Branch class or scalar')
        return res

    # add method right side
    def __radd__(self, left):
        res = self.copy()
        if isinstance(left, Branch):
            if res.dim != left.dim:
                raise ValueError('self.dim and left.dim must be equal')
            res.fvalue_y = res.fvalue_y + left.fvalue_y
        elif isscalar(left):
            res.fvalue_y = res.fvalue_y + left
        else:
            raise ValueError('left must be instance of Branch class or scalar')
        return res

    # sub method left side
    def __sub__(self, right):
        res = self.copy()
        if isinstance(right, Branch):
            if res.dim != right.dim:
                raise ValueError('self.dim and right.dim must be equal')
            res.fvalue_y = res.fvalue_y - right.fvalue_y
        elif isscalar(right):
            res.fvalue_y = res.fvalue_y - right
        else:
            raise ValueError('right must be instance of Branch class or scalar')
        return res

    # sub method right side
    def __rsub__(self, left):
        res = self.copy()
        if isinstance(left, Branch):
            if res.dim != left.dim:
                raise ValueError('self.dim and left.dim must be equal')
            res.fvalue_y = left.fvalue_y - res.fvalue_y
        elif isscalar(left):
            res.fvalue_y = left - res.fvalue_y
        else:
            raise ValueError('left must be instance of Branch class or scalar')
        return res

    # mul method left side
    def __mul__(self, right):
        res = self.copy()
        if isinstance(right, Branch):
            if res.dim != right.dim:
                raise ValueError('self.dim and right.dim must be equal')
            res.fvalue_y = res.fvalue_y * right.fvalue_y
        elif isscalar(right):
            res.fvalue_y = res.fvalue_y * right
        else:
            raise ValueError('right must be instance of Branch class or scalar')
        return res

    # mul method right side
    def __rmul__(self, left):
        res = self.copy()
        if isinstance(left, Branch):
            if res.dim != left.dim:
                raise ValueError('self.dim and left.dim must be equal')
            res.fvalue_y = res.fvalue_y * left.fvalue_y
        elif isscalar(left):
            res.fvalue_y = res.fvalue_y * left
        else:
            raise ValueError('left must be instance of Branch class or scalar')
        return res

    # div method left side
    def __truediv__(self, right):
        res = self.copy()
        if isinstance(right, Branch):
            if right.contains_zero():
                raise ZeroDivisionError('division by zero')
            if res.dim != right.dim:
                raise ValueError('self.dim and right.dim must be equal')
            res.fvalue_y = res.fvalue_y / right.fvalue_y
        elif isscalar(right):
            if right == 0.:
                raise ZeroDivisionError('division by zero')
            res.fvalue_y = res.fvalue_y / right
        else:
            raise ValueError('right must be instance of Branch class or scalar')
        return res

    # div method right side
    def __rtruediv__(self, left):
        if self.contains_zero():
            raise ZeroDivisionError('division by zero')
        res = self.copy()
        if isinstance(left, Branch):
            if res.dim != left.dim:
                raise ValueError('self.dim and left.dim must be equal')
            res.fvalue_y = left.fvalue_y / res.fvalue_y
        elif isscalar(left):
            res.fvalue_y = left / res.fvalue_y
        else:
            raise ValueError('left must be instance of Branch class or scalar')
        return res

    # neg method
    def __neg__(self):
        res = self.copy()
        res.fvalue_y = -res.fvalue_y
        return res

    # < method
    def __lt__(self, val):
        if isinstance(val, Branch):
            return np.max(self.fvalue_y) < np.min(val.fvalue_y)
        elif isscalar(val):
            return np.max(self.fvalue_y) < val
        else:
            raise ValueError('val must be instance of Branch class or scalar')

    # <= method
    def __le__(self, val):
        if isinstance(val, Branch):
            return np.max(self.fvalue_y) <= np.min(val.fvalue_y)
        elif isscalar(val):
            return np.max(self.fvalue_y) <= val
        else:
            raise ValueError('val must be instance of Branch class or scalar')

    # > method
    def __gt__(self, val):
        if isinstance(val, Branch):
            return np.min(self.fvalue_y) > np.max(val.fvalue_y)
        elif isscalar(val):
            return np.min(self.fvalue_y) > val
        else:
            raise ValueError('val must be instance of Branch class or scalar')

    # <= method
    def __ge__(self, val):
        if isinstance(val, Branch):
            return np.min(self.fvalue_y) >= np.max(val.fvalue_y)
        elif isscalar(val):
            return np.min(self.fvalue_y) >= val
        else:
            raise ValueError('val must be instance of Branch class or scalar')

    # == method
    def __eq__(self, val):
        if isinstance(val, Branch):
            return (self.fvalue_y == val.fvalue_y).all()
        elif isscalar(val):
            return (self.fvalue_y == val).all()
        else:
            raise ValueError('val must be instance of Branch class or scalar')

    # != method
    def __ne__(self, val):
        if isinstance(val, Branch):
            return (self.fvalue_y != val.fvalue_y).any()
        elif isscalar(val):
            return (self.fvalue_y != val).any()
        else:
            raise ValueError('val must be instance of Branch class or scalar')

    # alpha-cut method
    def acut(self, alpha):
        if alpha in self.domain_x:
            dom = self.domain_x.tolist()
            indx = dom.index(alpha)
            return self.fvalue_y[indx]
        elif alpha < 0.0 or alpha > 1.0:
            raise ValueError('alpha must be value between 0.0 and 1.0')
        else:
            for i in range(self.dim):
                if self.domain_x[i] > alpha:
                    xp = self.domain_x[i - 1]
                    xk = self.domain_x[i]
                    fp = self.fvalue_y[i - 1]
                    fk = self.fvalue_y[i]
                    factor = (alpha - xp) / (xk - xp)
                    return factor * fk + (1 - factor) * fp

    # to string method
    def __str__(self):
        return "\tdim: \t{0}\n\tdomain_x: {1}\n\tfvalue_y: {2}".format(self.dim, self.domain_x, self.fvalue_y)


# methods
# initialize branch of linear type y = a * x + b
def init_linear_ab(a, b, dim=11):
    if dim < 2:
        raise ValueError('dim must be > 1')
    branch = Branch(np.zeros(dim))
    branch.fvalue_y = a * branch.domain_x + b
    return branch


# initialize branch of linear type y = (x1 - x0) * x + x0
def init_linear_x0x1(x0, x1, dim=11):
    if dim < 2:
        raise ValueError('dim must be > 1')
    branch = Branch(np.zeros(dim))
    branch.fvalue_y = (x1 - x0) * branch.domain_x + x0
    return branch


# initialize branch of gaussian type y = s * sqrt(-2 * ln(x)) + m
def init_gauss(m, s, dim=11):
    if dim < 2:
        raise ValueError('dim must be > 1')
    branch = Branch(np.zeros(dim))
    branch.fvalue_y[1:] = s * np.sqrt(-2 * np.log(branch.domain_x[1:])) + m
    branch.fvalue_y[0] = s * np.sqrt(-2 * np.log(branch.domain_x[1] / 10)) + m
    #branch.fvalue_y[0] = s * np.sqrt(-2 * np.log(0.001)) + m
    return branch


def init_from_scalar(scalar, dim=11):
    if not isscalar(scalar):
        raise ValueError('scalar must be scalar')
    return Branch(np.ones(dim)*scalar)


def fabs(branch):
    res = branch.copy()
    res.fvalue_y = np.abs(res.fvalue_y)
    return res


def flog(branch):
    if branch <= 0.0:
        raise ValueError('branch.fvalue_y must be > 0')
    res = branch.copy()
    res.fvalue_y = np.log(res.fvalue_y)
    return res


def fexp(branch):
    res = branch.copy()
    res.fvalue_y = np.exp(res.fvalue_y)
    return res


def fpower(branch, p):
    res = branch.copy()
    if isinstance(p, Branch):
        if branch.dim != p.dim:
            raise ValueError('branch.dim and p.dim must be equal')
        res.fvalue_y = np.power(res.fvalue_y, p.fvalue_y)
    elif isscalar(p):
        res.fvalue_y = np.power(res.fvalue_y, p)
    return res


def fmax(branch1, branch2):
    if (not isinstance(branch1, Branch)) and (not isinstance(branch2, Branch)):
        raise ValueError('at least one of arguments must be a Branch')
    bran1 = init_from_scalar(branch1, dim=branch2.dim) if isscalar(branch1) else branch1.copy()
    bran2 = init_from_scalar(branch2, dim=branch1.dim) if isscalar(branch2) else branch2.copy()

    if bran1.dim != bran2.dim:
        raise ValueError('self.dim and right.dim must be equal')

    res = bran1.copy()
    res.fvalue_y = np.max([bran1.fvalue_y, bran2.fvalue_y], axis=0)
    return res


def fmin(branch1, branch2):
    if (not isinstance(branch1, Branch)) and (not isinstance(branch2, Branch)):
        raise ValueError('at least one of arguments must be a Branch')
    bran1 = init_from_scalar(branch1, dim=branch2.dim) if isscalar(branch1) else branch1.copy()
    bran2 = init_from_scalar(branch2, dim=branch1.dim) if isscalar(branch2) else branch2.copy()

    if bran1.dim != bran2.dim:
        raise ValueError('self.dim and right.dim must be equal')

    res = bran1.copy()
    res.fvalue_y = np.min([bran1.fvalue_y, bran2.fvalue_y], axis=0)
    return res
