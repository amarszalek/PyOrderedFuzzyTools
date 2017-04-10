//utils.c

#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

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
