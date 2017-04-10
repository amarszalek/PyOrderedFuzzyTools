//objective.c
#include "objective.h"
#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>


double obj_func_lin_reg(double *p, int np, double *xx, int nx, double *yy, int ny, int n_coef, int dim2, double *grad, int ng)
{
    int i, j, k, order, n_cans;
    double error;
    double *ar;

    error = dot_prod(p, p, np);


/*
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
                error += (cans[i*dim2+j]-ar[j])*(cans[i*dim2+j]-ar[j]);
                for(k=0; k<n_coef; k++)
                {
                    if(k==0)
                    {
                        grad[k*dim2+j] -= 2.0*(cans[i*dim2+j]-ar[j]);
                    }
                    else
                    {
                        grad[k*dim2+j] -= 2.0*(cans[i*dim2+j]-ar[j])*cans[(i-k)*dim2+j];
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
                error += (cans[i*dim2+j]-ar[j])*(cans[i*dim2+j]-ar[j]);
                for(k=0; k<n_coef; k++)
                {
                    grad[k*dim2+j] -= 2.0*(cans[i*dim2+j]-ar[j])*cans[(i-k-1)*dim2+j];
                }
            }
        }
    }
    free(ar);
    */
    return error;
}
