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
