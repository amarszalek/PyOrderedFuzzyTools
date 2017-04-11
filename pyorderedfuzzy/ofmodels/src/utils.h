//utils.h

#ifndef utils_h
#define utils_h

void lin_reg(double *p, double *xx, int n_coef, int dim2, int s, double *pred);
void ar_bias(double *p, double *past, int order, int dim2, int s, double *ar);
void ar_unbias(double *p, double *past, int order, int dim2, int s, double *ar);
double dot_prod(double *x, double *y, int n);


#endif