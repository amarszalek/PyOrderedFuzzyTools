//utils.c

#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void lin_reg(double *p, double *xx, int n_coef, int dim2, int s, double *pred)
{
    int i, k;

    for(i=0;i<dim2;i++) pred[i] = 0.0;

    for(k=0;k<n_coef;k++)
    {
        for(i=0;i<dim2;i++)
        {
            pred[i]+=p[k*dim2+i]*xx[s+k*dim2+i];
        }
    }
}


double dot_prod(double *x, double *y, int n)
{
    int i;
    double suma;
    suma = 0.0;
    for(i=0;i<n;i++)
    {
        suma += x[i]*y[i];
    }
    return suma;
}
