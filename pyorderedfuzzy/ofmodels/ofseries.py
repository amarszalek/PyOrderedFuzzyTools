# -*- coding: utf-8 -*-

import numpy as np
import copy
import statsmodels.api as sm
from pyorderedfuzzy.ofnumbers.ofnumber import OFNumber, flog
from pyorderedfuzzy.ofcandles.ofcandle import OFCandle

__author__ = "amarszalek"


class OFSeries(list):
    def __init__(self, ofns):
        list.__init__(self, ofns)

    def copy(self):
        return copy.deepcopy(self)

    def transform(self, method='diff'):
        arr = np.array(self, dtype=object)
        if method == 'diff':
            new_ofns = arr[1:]-arr[:-1]
        elif method == 'ret':
            new_ofns = (arr[1:]-arr[:-1])/arr[:-1]
        elif method == 'logret':
            new_ofns = map(flog, (arr[1:]/arr[:-1]))
        else:
            raise ValueError('method must be diff, ret or logret')
        return OFSeries(new_ofns)

    # Augmented Dickey-Fuller test from statsmodels.tsa
    def adfuller(self, verbose='max', s=0, e=None, **kwargs):
        if e is None:
            ofns = self[s:]
        else:
            ofns = self[s:e]
        n = len(ofns)
        dim = ofns[0].branch_f.dim
        res = []
        for i in range(dim):
            series = np.array([ofns[j].branch_f.fvalue_y[i] for j in range(n)])
            r = sm.tsa.adfuller(series, **kwargs)
            if verbose == 'full':
                print('f:', i, '  statistic:', r[0], '  pvalue:', r[1], '  1%:', r[4]['1%'], '  5%:', r[4]['5%'])
            res.append(r)
        dim = ofns[0].branch_g.dim
        for i in range(dim):
            series = np.array([ofns[j].branch_g.fvalue_y[i] for j in range(n)])
            r = sm.tsa.adfuller(series, **kwargs)
            if verbose == 'full':
                print('g:', i, '  statistic:', r[0], '  pvalue:', r[1], '  1%:', r[4]['1%'], '  5%:', r[4]['5%'])
            res.append(r)
        if verbose in ['max', 'full']:
            a = np.array([r[0] for r in res])
            print('max of DF-statistics:', np.max(a), '  1%:', res[0][4]['1%'], '  5%:', res[0][4]['5%'])
        return res

    # Partial autocorrelation estimated from statsmodels.tsa
    def pacf(self, nlags=20, method='ywunbiased', alpha=None, verbose='maxmin', s=0, e=None):
        if e is None:
            ofns = self[s:]
        else:
            ofns = self[s:e]
        n = len(ofns)
        dim = ofns[0].branch_f.dim
        res = []
        for i in range(dim):
            series = np.array([ofns[j].branch_f.fvalue_y[i] for j in range(n)])
            r = sm.tsa.pacf(series, nlags=nlags, method=method, alpha=alpha)
            if verbose == 'full':
                print('f:', i)
                print('pacf:', r)
            res.append(r)
        dim = ofns[0].branch_g.dim
        for i in range(dim):
            series = np.array([ofns[j].branch_g.fvalue_y[i] for j in range(n)])
            r = sm.tsa.pacf(series, nlags=nlags, method=method, alpha=alpha)
            if verbose == 'full':
                print('g:', i)
                print('pacf:', r)
            res.append(r)
        if verbose in ['maxmin', 'full']:
            a = np.array(res)
            print('max:', np.max(a, axis=0))
            print('min:', np.min(a, axis=0))
        return res

    # Returns information criteria for many ARMA models, arma_order_selctic from statsmodels.tsa
    def arma_select_order_ic(self, max_ar=5, max_ma=0, ic=['aic'], trend='c', verbose='max', s=0, e=None, **kwargs):
        if e is None:
            ofns = self[s:]
        else:
            ofns = self[s:e]
        n = len(ofns)
        dim = ofns[0].branch_f.dim
        res = []
        for i in range(dim):
            series = np.array([ofns[j].branch_f.fvalue_y[i] for j in range(n)])
            r = sm.tsa.arma_order_select_ic(series, max_ar=max_ar, max_ma=max_ma, ic=ic, trend=trend, **kwargs)
            if verbose == 'full':
                print('f:', i)
                for c in ic:
                    print(c, ':', r[str(c)+'_min_order'])
            res.append(r)
        dim = ofns[0].branch_g.dim
        for i in range(dim):
            series = np.array([ofns[j].branch_g.fvalue_y[i] for j in range(n)])
            r = sm.tsa.arma_order_select_ic(series, max_ar=max_ar, max_ma=max_ma, ic=ic, trend=trend, **kwargs)
            if verbose == 'full':
                print('g:', i)
                for c in ic:
                    print(c, ':', r[str(c)+'_min_order'])
            res.append(r)
        if verbose in ['max', 'full']:
            for c in ic:
                print('max '+str(c)+'_min_order:', np.max(np.array([res[i][str(c)+'_min_order']
                                                                    for i in range(len(res))])))
        return res

    def plot_ofseries(self, ax, s=0, e=None, color='black', shift=0, ord_method='expected'):
        if e is None:
            ofns = self[s:]
        else:
            ofns = self[s:e]
        n = len(ofns)
        for i in range(n):
            f = ofns[i].branch_f
            g = ofns[i].branch_g
            fg_y = np.append(f.fvalue_y,  g.fvalue_y[::-1])
            fg_x = np.append(f.domain_x, g.domain_x[::-1])*0.5 + i-0.25 + shift
            o = ofns[i].order(method=ord_method)
            if o >= 0:
                ax.fill_betweenx(fg_y, i+shift-0.25, fg_x, facecolor='white', edgecolor='black')
            else:
                ax.fill_betweenx(fg_y, i+shift-0.25, fg_x, facecolor=color, edgecolor='black')
            ax.plot([i+shift-0.25, i+shift-0.25], [np.min(fg_y), np.max(fg_y)], c='w', lw=2)
            ax.plot([i+shift-0.25, i+shift-0.25], [f.fvalue_y[0], g.fvalue_y[0]], c='black', lw=2)
        ax.set_xlim(-1+shift, n+shift)

    def histogram(self, level='f_0', bins=10, normed=False, s=0, e=None, **kwargs):
        lev = level.split('_')
        branch = lev[0]
        alpha = float(lev[1])
        if e is None:
            e = len(self)
        if 0.0 <= alpha <= 1.0:
            if branch == 'f':
                data = np.array([self[i].branch_f.acut(alpha) for i in range(s, e)])
            elif branch == 'g':
                data = np.array([self[i].branch_g.acut(alpha) for i in range(s, e)])
            else:
                raise ValueError('wrong level syntax')
        else:
            raise ValueError('wrong level synatx')
        return np.histogram(data, bins=bins, normed=normed, **kwargs), data


def init_from_stock_data(ctype, data, params):
    ofcans = []
    for i in range(len(data['price'])):
        c_data = {'price': data['price'][i], 'volume': data['volume'][i], 'spread': data['spread'][i]}
        ofcans.append(OFCandle(ctype, c_data, params).to_ofn())
    return OFSeries(ofcans)


def init_from_scalar_values(prices, dim):
    ofns = [OFNumber(np.ones(dim)*p, np.ones(dim)*p) for p in prices]
    return OFSeries(ofns)
