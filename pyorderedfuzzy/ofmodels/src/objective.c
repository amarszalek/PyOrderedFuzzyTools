//objective.c
#include "objective.h"
#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>


double obj_func_lin_reg(double *p, int np, double *xx, int nx, double *yy, int ny, int n_coef, int dim2, double *grad, int ng)
{
    int t, j, k, T;
    double r, error;
    double *pred;

    T = (int)(ny/dim2);
    pred = (double*)malloc(dim2*sizeof(double));

    for(j=0;j<ng;j++) grad[j]=0.0;

    error = 0.0;
    for(t=0; t<T; t++)
    {
        lin_reg(p, xx, n_coef, dim2, t*n_coef*dim2, pred);
        for(j=0;j<dim2;j++)
        {
            r = yy[t*dim2+j] - pred[j];
            error += r*r;
            for(k=0; k<n_coef; k++)
            {
                grad[k*dim2+j] -= 2.0*r*xx[t*n_coef*dim2+k*dim2+j];
            }
        }
    }

    free(pred);
    return error;
}


double obj_func_ar_ls(double *p, int np, double *cans, int nc, int n_coef, int dim2, int intercept, double *grad, int ng)
{
    int i, j, k, order, n_cans;
    double error, r;
    double *ar;

    for(j=0;j<ng;j++) grad[j]=0.0;

    n_cans = (int)(nc/dim2);
    ar=(double*)malloc(dim2*sizeof(double));
    order = n_coef;
    if(intercept == 1) order = n_coef - 1;

    error = 0.0;
    if(intercept == 1)
    {
        for(i=order; i<n_cans; i++)
        {
            ar_bias(p, cans, order, dim2, i*dim2, ar);
            for(j=0;j<dim2;j++)
            {
                r = cans[i*dim2+j]-ar[j];
                error += r*r;
                for(k=0; k<n_coef; k++)
                {
                    if(k==0)
                    {
                        grad[k*dim2+j] -= 2.0*r;
                    }
                    else
                    {
                        grad[k*dim2+j] -= 2.0*r*cans[(i-k)*dim2+j];
                    }
                }
            }
        }
    }
    else
    {
        for(i=order; i<n_cans; i++)
        {
            ar_unbias(p, cans, order, dim2, i*dim2, ar);
            for(j=0;j<dim2;j++)
            {
                r = cans[i*dim2+j]-ar[j];
                error += r*r;
                for(k=0; k<n_coef; k++)
                {
                    grad[k*dim2+j] -= 2.0*r*cans[(i-k-1)*dim2+j];
                }
            }
        }
    }
    free(ar);
    return error;
}