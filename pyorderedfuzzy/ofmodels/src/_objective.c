//_objective.c

#include <Python.h>
#include <numpy/arrayobject.h>
#include "objective.h"

static char module_docstring[] = "This module provides an interface for calculating objectives using C.";
static char obj_func_lin_reg_docstring[] = "Calculate objective function for linear regression.";
static char obj_func_ar_ls_docstring[] = "Calculate objective function (ls) for auto-regressive.";

static PyObject *objective_obj_func_lin_reg(PyObject *self, PyObject *args);
static PyObject *objective_obj_func_ar_ls(PyObject *self, PyObject *args);

static PyMethodDef module_methods[] = {
    {"obj_func_lin_reg", objective_obj_func_lin_reg, METH_VARARGS, obj_func_lin_reg_docstring},
    {"obj_func_ar_ls", objective_obj_func_ar_ls, METH_VARARGS, obj_func_ar_ls_docstring},
    {NULL, NULL, 0, NULL}
};


PyMODINIT_FUNC PyInit__objective(void)
{
    PyObject *module;
    static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "_objective",
        module_docstring,
        -1,
        module_methods,
        NULL,
        NULL,
        NULL,
        NULL
    };
    module = PyModule_Create(&moduledef);
    if (!module) return NULL;

    /* Load `numpy` functionality. */
    import_array();

    return module;
}

static PyObject *objective_obj_func_lin_reg(PyObject *self, PyObject *args)
{
    int i, n_coef, dim2;
    PyObject *p_obj, *xx_obj, *yy_obj;

    /* Parse the input tuple */
    if (!PyArg_ParseTuple(args, "OOOii", &p_obj, &xx_obj, &yy_obj, &n_coef, &dim2))
        return NULL;

    /* Interpret the input objects as numpy arrays. */
    PyObject *p_array = PyArray_FROM_OTF(p_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *xx_array = PyArray_FROM_OTF(xx_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *yy_array = PyArray_FROM_OTF(yy_obj, NPY_DOUBLE, NPY_IN_ARRAY);

    /* If that didn't work, throw an exception. */
    if (p_array == NULL || xx_array == NULL || yy_array == NULL ) {
        Py_XDECREF(p_array);
        Py_XDECREF(xx_array);
        Py_XDECREF(yy_array);
        return NULL;
    }

    /* How many data points are there? */
    int np = (int)PyArray_DIM(p_array, 0);
    int nx = (int)PyArray_DIM(xx_array, 0);
    int ny = (int)PyArray_DIM(yy_array, 0);
    int ng = n_coef*dim2;

    /* Get pointers to the data as C-types. */
    double *p = (double*)PyArray_DATA(p_array);
    double *xx = (double*)PyArray_DATA(xx_array);
    double *yy = (double*)PyArray_DATA(yy_array);
    double *g;

    g=(double*)malloc(ng*sizeof(double));

    /* Call the external C function to compute the chi-squared. */
    double error = obj_func_lin_reg(p, np, xx, nx, yy, ny, n_coef, dim2, g, ng);

    /* Clean up. */
    Py_DECREF(p_array);
    Py_DECREF(xx_array);
    Py_DECREF(yy_array);

    /* Build the output tuple */


    PyObject *grad = PyList_New(ng);
    for(i=0;i<ng;i++) PyList_SET_ITEM(grad, i, Py_BuildValue("d", g[i]));

    PyObject *ret = PyTuple_New(2);
    PyTuple_SET_ITEM(ret, 0, Py_BuildValue("d", error));
    PyTuple_SET_ITEM(ret, 1, Py_BuildValue("O", grad));
    //PyObject *ret = Py_BuildValue("d", error);
    free(g);
    return ret;
}


static PyObject *objective_obj_func_ar_ls(PyObject *self, PyObject *args)
{
    int i, n_coef, dim2, intercept;
    PyObject *p_obj, *ofns_obj;

    /* Parse the input tuple */
    if (!PyArg_ParseTuple(args, "OOiii", &p_obj, &ofns_obj, &n_coef, &dim2, &intercept))
        return NULL;

    /* Interpret the input objects as numpy arrays. */
    PyObject *p_array = PyArray_FROM_OTF(p_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *ofns_array = PyArray_FROM_OTF(ofns_obj, NPY_DOUBLE, NPY_IN_ARRAY);

    /* If that didn't work, throw an exception. */
    if (p_array == NULL || ofns_array == NULL ) {
        Py_XDECREF(p_array);
        Py_XDECREF(ofns_array);
        return NULL;
    }

    /* How many data points are there? */
    int np = (int)PyArray_DIM(p_array, 0);
    int nc = (int)PyArray_DIM(ofns_array, 0);
    int ng = n_coef*dim2;

    /* Get pointers to the data as C-types. */
    double *p = (double*)PyArray_DATA(p_array);
    double *ofns = (double*)PyArray_DATA(ofns_array);
    double *g;

    g=(double*)malloc(ng*sizeof(double));

    /* Call the external C function to compute the chi-squared. */
    double error = obj_func_ar_ls(p, np, ofns, nc, n_coef, dim2, intercept, g, ng);

    /* Clean up. */
    Py_DECREF(p_array);
    Py_DECREF(ofns_array);

    /* Build the output tuple */
    PyObject *grad = PyList_New(ng);
    for(i=0;i<ng;i++) PyList_SET_ITEM(grad, i, Py_BuildValue("d", g[i]));

    PyObject *ret = PyTuple_New(2);
    PyTuple_SET_ITEM(ret, 0, Py_BuildValue("d", error));
    PyTuple_SET_ITEM(ret, 1, Py_BuildValue("O", grad));
    //PyObject *ret = Py_BuildValue("d", error);
    free(g);
    return ret;
}