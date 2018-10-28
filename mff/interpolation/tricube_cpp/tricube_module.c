#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <structmember.h>
#include <numpy/arrayobject.h>

#include "_tricube.h"


static PyObject* reg_ev_energy_wrapper(PyObject* self, PyObject* args)
{
    PyObject *ret = NULL;
    PyObject *x0 = NULL;
    PyObject *x1 = NULL;
    PyObject *x2 = NULL;
    PyObject *f = NULL;
    PyObject *fx0 = NULL;
    PyObject *fx1 = NULL;
    PyObject *fx2 = NULL;
    PyArrayObject *x0_array = NULL;
    PyArrayObject *x1_array = NULL;
    PyArrayObject *x2_array = NULL;
    PyArrayObject *f_array = NULL;
    PyArrayObject *fx0_array = NULL;
    PyArrayObject *fx1_array = NULL;
    PyArrayObject *fx2_array = NULL;
    PyArrayObject *result_array = NULL;
    npy_intp *shape_x0, *shape_x1, *shape_x2, *shape_f;

    if (!PyArg_ParseTuple(args, "OOOOOOO", &x0, &x1, &x2, &f, &fx0, &fx1, &fx2)) {
        Py_INCREF(Py_None);
        return Py_None;
    }
    x0_array = (PyArrayObject *) PyArray_FROM_OTF(x0, NPY_FLOAT64, NPY_ARRAY_IN_ARRAY);
    if (x0_array == NULL) {
        goto out;
    }
    //if (PyArray_NDIM(x0_array) != 1) {
    //    PyErr_SetString(PyExc_ValueError, "Dim x0 != 1");
    //    goto out;
    //}
    x1_array = (PyArrayObject *) PyArray_FROM_OTF(x1, NPY_FLOAT64, NPY_ARRAY_IN_ARRAY);
    if (x1_array == NULL) {
        goto out;
    }

    x2_array = (PyArrayObject *) PyArray_FROM_OTF(x2, NPY_FLOAT64, NPY_ARRAY_IN_ARRAY);
    if (x2_array == NULL) {
        goto out;
    }

    f_array = (PyArrayObject *) PyArray_FROM_OTF(f, NPY_FLOAT64, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    if (f_array == NULL) {
        goto out;
    }

    fx0_array = (PyArrayObject *) PyArray_FROM_OTF(fx0, NPY_FLOAT64, NPY_ARRAY_IN_ARRAY);
    if (fx0_array == NULL) {
        goto out;
    }

    fx1_array = (PyArrayObject *) PyArray_FROM_OTF(fx1, NPY_FLOAT64, NPY_ARRAY_IN_ARRAY);
    if (fx1_array == NULL) {
        goto out;
    }

    fx2_array = (PyArrayObject *) PyArray_FROM_OTF(fx2, NPY_FLOAT64, NPY_ARRAY_IN_ARRAY);
    if (x2_array == NULL) {
        goto out;
    }


    shape_x0 = PyArray_DIMS(x0_array);
    shape_x1 = PyArray_DIMS(x1_array);
    shape_x2 = PyArray_DIMS(x2_array);
    shape_f = PyArray_DIMS(f_array);


    /* Allocate the result array */
    result_array = (PyArrayObject *) PyArray_SimpleNew(1, shape_x0, NPY_FLOAT64);
    if (result_array == NULL) {
        goto out;
    }

    /* Now we can run the real calculation using pointers to the memory that
     * numpy has allocated */

    reg_ev_energy(
        (double *) PyArray_DATA(result_array),
        (double *) PyArray_DATA(x0_array), (double *) PyArray_DATA(x1_array), (double *) PyArray_DATA(x2_array),
        (double *) PyArray_DATA(f_array),
        (double *) PyArray_DATA(fx0_array), (double *) PyArray_DATA(fx1_array), (double *) PyArray_DATA(fx2_array),
        (int) shape_f[0], (int) shape_f[1], (int) shape_f[2], (int) shape_x0[0]);


    ret = (PyObject *) result_array;
    result_array = NULL;

out:
    Py_XDECREF(x0_array);
    Py_XDECREF(x1_array);
    Py_XDECREF(x2_array);
    Py_XDECREF(f_array);
    Py_XDECREF(fx0_array);
    Py_XDECREF(fx1_array);
    Py_XDECREF(fx2_array);
    Py_XDECREF(result_array);
    return ret;
}

static PyObject* reg_ev_forces_wrapper(PyObject* self, PyObject* args)
{
    //PyObject *ret = NULL;
    PyObject * volatile capi_buildvalue = NULL;
    //PyArrayObject *capi_val_dx0_tmp = NULL;
    //PyArrayObject *capi_val_dx1_tmp = NULL;
    //PyArrayObject *capi_val_dx2_tmp = NULL;

    PyObject *x0 = NULL;
    PyObject *x1 = NULL;
    PyObject *x2 = NULL;
    PyObject *f = NULL;
    PyObject *fx0 = NULL;
    PyObject *fx1 = NULL;
    PyObject *fx2 = NULL;
    PyArrayObject *x0_array = NULL;
    PyArrayObject *x1_array = NULL;
    PyArrayObject *x2_array = NULL;
    PyArrayObject *f_array = NULL;
    PyArrayObject *fx0_array = NULL;
    PyArrayObject *fx1_array = NULL;
    PyArrayObject *fx2_array = NULL;

    PyArrayObject *result_dx0 = NULL;
    PyArrayObject *result_dx1 = NULL;
    PyArrayObject *result_dx2 = NULL;
    npy_intp *shape_x0, *shape_x1, *shape_x2, *shape_f;

    if (!PyArg_ParseTuple(args, "OOOOOOO", &x0, &x1, &x2, &f, &fx0, &fx1, &fx2)) {
        Py_INCREF(Py_None);
        return Py_None;
    }
    x0_array = (PyArrayObject *) PyArray_FROM_OTF(x0, NPY_FLOAT64, NPY_ARRAY_IN_ARRAY);
    if (x0_array == NULL) {
        goto out;
    }
    //if (PyArray_NDIM(x0_array) != 1) {
    //    PyErr_SetString(PyExc_ValueError, "Dim x0 != 1");
    //    goto out;
    //}
    x1_array = (PyArrayObject *) PyArray_FROM_OTF(x1, NPY_FLOAT64, NPY_ARRAY_IN_ARRAY);
    if (x1_array == NULL) {
        goto out;
    }

    x2_array = (PyArrayObject *) PyArray_FROM_OTF(x2, NPY_FLOAT64, NPY_ARRAY_IN_ARRAY);
    if (x2_array == NULL) {
        goto out;
    }

    f_array = (PyArrayObject *) PyArray_FROM_OTF(f, NPY_FLOAT64, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    if (f_array == NULL) {
        goto out;
    }

    fx0_array = (PyArrayObject *) PyArray_FROM_OTF(fx0, NPY_FLOAT64, NPY_ARRAY_IN_ARRAY);
    if (fx0_array == NULL) {
        goto out;
    }

    fx1_array = (PyArrayObject *) PyArray_FROM_OTF(fx1, NPY_FLOAT64, NPY_ARRAY_IN_ARRAY);
    if (fx1_array == NULL) {
        goto out;
    }

    fx2_array = (PyArrayObject *) PyArray_FROM_OTF(fx2, NPY_FLOAT64, NPY_ARRAY_IN_ARRAY);
    if (x2_array == NULL) {
        goto out;
    }


    shape_x0 = PyArray_DIMS(x0_array);
    shape_x1 = PyArray_DIMS(x1_array);
    shape_x2 = PyArray_DIMS(x2_array);
    shape_f = PyArray_DIMS(f_array);


    /* Allocate the result array */
    result_dx0 = (PyArrayObject *) PyArray_SimpleNew(1, shape_x0, NPY_FLOAT64);
    if (result_dx0 == NULL) {
        goto out;
    }
    result_dx1 = (PyArrayObject *) PyArray_SimpleNew(1, shape_x0, NPY_FLOAT64);
    if (result_dx1 == NULL) {
        goto out;
    }
    result_dx2 = (PyArrayObject *) PyArray_SimpleNew(1, shape_x0, NPY_FLOAT64);
    if (result_dx2 == NULL) {
        goto out;
    }

    /* Now we can run the real calculation using pointers to the memory that
     * numpy has allocated */

    reg_ev_forces(
        (double *) PyArray_DATA(result_dx0), (double *) PyArray_DATA(result_dx1), (double *) PyArray_DATA(result_dx2),
        (double *) PyArray_DATA(x0_array), (double *) PyArray_DATA(x1_array), (double *) PyArray_DATA(x2_array),
        (double *) PyArray_DATA(f_array),
        (double *) PyArray_DATA(fx0_array), (double *) PyArray_DATA(fx1_array), (double *) PyArray_DATA(fx2_array),
        (int) shape_f[0], (int) shape_f[1], (int) shape_f[2], (int) shape_x0[0]);


    capi_buildvalue = Py_BuildValue("OOO", result_dx0, result_dx1, result_dx2);
//    capi_val_dx0_tmp = (PyObject *) result_dx0;
//    capi_val_dx1_tmp = (PyObject *) result_dx1;
//    capi_val_dx2_tmp = (PyObject *) result_dx2;
    result_dx0 = NULL;
    result_dx1 = NULL;
    result_dx2 = NULL;



out:
    Py_XDECREF(x0_array);
    Py_XDECREF(x1_array);
    Py_XDECREF(x2_array);
    Py_XDECREF(f_array);
    Py_XDECREF(fx0_array);
    Py_XDECREF(fx1_array);
    Py_XDECREF(fx2_array);
    Py_XDECREF(result_dx0);
    Py_XDECREF(result_dx1);
    Py_XDECREF(result_dx2);
    return capi_buildvalue;
}

static PyObject* reg_ev_all_wrapper(PyObject* self, PyObject* args)
{
    //PyObject *ret = NULL;
    PyObject * volatile capi_buildvalue = NULL;
    //PyArrayObject *capi_val_dx0_tmp = NULL;
    //PyArrayObject *capi_val_dx1_tmp = NULL;
    //PyArrayObject *capi_val_dx2_tmp = NULL;

    PyObject *x0 = NULL;
    PyObject *x1 = NULL;
    PyObject *x2 = NULL;
    PyObject *f = NULL;
    PyObject *fx0 = NULL;
    PyObject *fx1 = NULL;
    PyObject *fx2 = NULL;
    PyArrayObject *x0_array = NULL;
    PyArrayObject *x1_array = NULL;
    PyArrayObject *x2_array = NULL;
    PyArrayObject *f_array = NULL;
    PyArrayObject *fx0_array = NULL;
    PyArrayObject *fx1_array = NULL;
    PyArrayObject *fx2_array = NULL;

    PyArrayObject *result_array = NULL;
    PyArrayObject *result_dx0 = NULL;
    PyArrayObject *result_dx1 = NULL;
    PyArrayObject *result_dx2 = NULL;
    npy_intp *shape_x0, *shape_x1, *shape_x2, *shape_f;

    if (!PyArg_ParseTuple(args, "OOOOOOO", &x0, &x1, &x2, &f, &fx0, &fx1, &fx2)) {
        Py_INCREF(Py_None);
        return Py_None;
    }
    x0_array = (PyArrayObject *) PyArray_FROM_OTF(x0, NPY_FLOAT64, NPY_ARRAY_IN_ARRAY);
    if (x0_array == NULL) {
        goto out;
    }
    //if (PyArray_NDIM(x0_array) != 1) {
    //    PyErr_SetString(PyExc_ValueError, "Dim x0 != 1");
    //    goto out;
    //}
    x1_array = (PyArrayObject *) PyArray_FROM_OTF(x1, NPY_FLOAT64, NPY_ARRAY_IN_ARRAY);
    if (x1_array == NULL) {
        goto out;
    }

    x2_array = (PyArrayObject *) PyArray_FROM_OTF(x2, NPY_FLOAT64, NPY_ARRAY_IN_ARRAY);
    if (x2_array == NULL) {
        goto out;
    }

    f_array = (PyArrayObject *) PyArray_FROM_OTF(f, NPY_FLOAT64, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    if (f_array == NULL) {
        goto out;
    }

    fx0_array = (PyArrayObject *) PyArray_FROM_OTF(fx0, NPY_FLOAT64, NPY_ARRAY_IN_ARRAY);
    if (fx0_array == NULL) {
        goto out;
    }

    fx1_array = (PyArrayObject *) PyArray_FROM_OTF(fx1, NPY_FLOAT64, NPY_ARRAY_IN_ARRAY);
    if (fx1_array == NULL) {
        goto out;
    }

    fx2_array = (PyArrayObject *) PyArray_FROM_OTF(fx2, NPY_FLOAT64, NPY_ARRAY_IN_ARRAY);
    if (x2_array == NULL) {
        goto out;
    }


    shape_x0 = PyArray_DIMS(x0_array);
    shape_x1 = PyArray_DIMS(x1_array);
    shape_x2 = PyArray_DIMS(x2_array);
    shape_f = PyArray_DIMS(f_array);


    /* Allocate the result array */
    result_array = (PyArrayObject *) PyArray_SimpleNew(1, shape_x0, NPY_FLOAT64);
    if (result_array == NULL) {
        goto out;
    }

    result_dx0 = (PyArrayObject *) PyArray_SimpleNew(1, shape_x0, NPY_FLOAT64);
    if (result_dx0 == NULL) {
        goto out;
    }
    result_dx1 = (PyArrayObject *) PyArray_SimpleNew(1, shape_x0, NPY_FLOAT64);
    if (result_dx1 == NULL) {
        goto out;
    }
    result_dx2 = (PyArrayObject *) PyArray_SimpleNew(1, shape_x0, NPY_FLOAT64);
    if (result_dx2 == NULL) {
        goto out;
    }

    /* Now we can run the real calculation using pointers to the memory that
     * numpy has allocated */

    reg_ev_all(
        (double *) PyArray_DATA(result_array),
        (double *) PyArray_DATA(result_dx0), (double *) PyArray_DATA(result_dx1), (double *) PyArray_DATA(result_dx2),
        (double *) PyArray_DATA(x0_array), (double *) PyArray_DATA(x1_array), (double *) PyArray_DATA(x2_array),
        (double *) PyArray_DATA(f_array),
        (double *) PyArray_DATA(fx0_array), (double *) PyArray_DATA(fx1_array), (double *) PyArray_DATA(fx2_array),
        (int) shape_f[0], (int) shape_f[1], (int) shape_f[2], (int) shape_x0[0]);


    capi_buildvalue = Py_BuildValue("OOOO", result_array, result_dx0, result_dx1, result_dx2);
//    capi_val_dx0_tmp = (PyObject *) result_dx0;
//    capi_val_dx1_tmp = (PyObject *) result_dx1;
//    capi_val_dx2_tmp = (PyObject *) result_dx2;
    result_array = NULL;
    result_dx0 = NULL;
    result_dx1 = NULL;
    result_dx2 = NULL;



out:
    Py_XDECREF(x0_array);
    Py_XDECREF(x1_array);
    Py_XDECREF(x2_array);
    Py_XDECREF(f_array);
    Py_XDECREF(fx0_array);
    Py_XDECREF(fx1_array);
    Py_XDECREF(fx2_array);
    Py_XDECREF(result_array);
    Py_XDECREF(result_dx0);
    Py_XDECREF(result_dx1);
    Py_XDECREF(result_dx2);

    return capi_buildvalue;
}


static PyMethodDef tricube_methods[] = {
    {"reg_ev_energy", (PyCFunction) reg_ev_energy_wrapper, METH_VARARGS,
         "Runs an algorithm defined in a local C file."},
    {"reg_ev_forces", (PyCFunction) reg_ev_forces_wrapper, METH_VARARGS,
         "Runs an algorithm defined in a local C file."},
    {"reg_ev_all", (PyCFunction) reg_ev_all_wrapper, METH_VARARGS,
         "Runs an algorithm defined in a local C file."},
    {NULL, NULL, 0, NULL}  /* sentinel */
};

static struct PyModuleDef tricube_module = {
    PyModuleDef_HEAD_INIT,
    "_tricube",
    NULL,
    -1,
    tricube_methods,
    NULL,
    NULL,
    NULL,
    NULL
};

PyMODINIT_FUNC PyInit__tricube(void)
{
    PyObject *mod = NULL;


    mod = PyModule_Create(&tricube_module);
    if (!mod) {
        return NULL;
    }

    import_array();

    return mod;
}


