//objective.h

#ifndef objective_h
#define objective_h

double obj_func_lin_reg(double *p, int np, double *xx, int nx, double *yy, int ny, int n_coef, int dim2, double *grad, int ng);
double obj_func_ar_ls(double *p, int np, double *cans, int nc, int n_coef, int dim2, int intercept, double *grad, int ng);
double obj_func_ar_cml(double *p, int np, double *cans, int nc, int n_coef, int dim2, int intercept, double *grad, int ng);

#endif
