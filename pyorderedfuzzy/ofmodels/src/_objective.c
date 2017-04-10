//_objective.c

#include <Python.h>
#include <numpy/arrayobject.h>
#include "objective.h"

static char module_docstring[] = "This module provides an interface for calculating objectives using C.";
static char obj_func_lin_reg_docstring[] = "Calculate objective function for linear regression.";

static PyObject *objective_obj_func_lin_reg(PyObject *self, PyObject *args);

static PyMethodDef module_methods[] = {
    {"objective", objective_obj_func_lin_reg, METH_VARARGS, obj_func_lin_reg_docstring},
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
    int np, nx, ny, ng, n_coef, dim2;
    PyObject *p_obj, *xx_obj, *yy_obj, *g_obj;

    /* Parse the input tuple */
    if (!PyArg_ParseTuple(args, "ddOOO", &p_obj, &xx_obj, &yy_obj, &n_coef, &dim2, &g_obj))
        return NULL;

    /* Interpret the input objects as numpy arrays. */
    PyObject *p_array = PyArray_FROM_OTF(p_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *xx_array = PyArray_FROM_OTF(xx_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *yy_array = PyArray_FROM_OTF(yy_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *g_array = PyArray_FROM_OTF(g_obj, NPY_DOUBLE, NPY_IN_ARRAY);

    /* If that didn't work, throw an exception. */
    if (p_array == NULL || xx_array == NULL || yy_array == NULL || yy_array == NULL) {
        Py_XDECREF(p_array);
        Py_XDECREF(xx_array);
        Py_XDECREF(yy_array);
        Py_XDECREF(g_array);
        return NULL;
    }

    /* How many data points are there? */
    int np = (int)PyArray_DIM(p_array, 0);
    int nx = (int)PyArray_DIM(xx_array, 0);
    int ny = (int)PyArray_DIM(yy_array, 0);
    int ng = (int)PyArray_DIM(g_array, 0);

    /* Get pointers to the data as C-types. */
    double *p    = (double*)PyArray_DATA(p_array);
    double *xx    = (double*)PyArray_DATA(xx_array);
    double *yy = (double*)PyArray_DATA(yy_array);
    double *g = (double*)PyArray_DATA(g_array);

    /* Call the external C function to compute the chi-squared. */
    double value = obj_func_lin_reg(p, np, xx, nx, yy, ny, n_coef, dim2, g, ng);

    /* Clean up. */
    Py_DECREF(p_array);
    Py_DECREF(xx_array);
    Py_DECREF(yy_array);
    Py_DECREF(g_array);


    /* Build the output tuple */
    PyObject *ret = Py_BuildValue("d", value);
    return ret;
}