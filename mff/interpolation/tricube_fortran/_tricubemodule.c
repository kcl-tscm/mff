/* File: _tricubemodule.c
 * This file is auto-generated with f2py (version:2).
 * f2py is a Fortran to Python Interface Generator (FPIG), Second Edition,
 * written by Pearu Peterson <pearu@cens.ioc.ee>.
 * Generation date: Tue Jun 12 17:29:17 2018
 * Do not edit this file directly unless you know what you are doing!!!
 */

#ifdef __cplusplus
extern "C" {
#endif

/*********************** See f2py2e/cfuncs.py: includes ***********************/
#include "Python.h"
#include <stdarg.h>
#include "fortranobject.h"
#include <math.h>

/**************** See f2py2e/rules.py: mod_rules['modulebody'] ****************/
static PyObject *_tricube_error;
static PyObject *_tricube_module;

/*********************** See f2py2e/cfuncs.py: typedefs ***********************/
/*need_typedefs*/

/****************** See f2py2e/cfuncs.py: typedefs_generated ******************/
/*need_typedefs_generated*/

/********************** See f2py2e/cfuncs.py: cppmacros **********************/
#define rank(var) var ## _Rank
#define shape(var,dim) var ## _Dims[dim]
#define old_rank(var) (PyArray_NDIM((PyArrayObject *)(capi_ ## var ## _tmp)))
#define old_shape(var,dim) PyArray_DIM(((PyArrayObject *)(capi_ ## var ## _tmp)),dim)
#define fshape(var,dim) shape(var,rank(var)-dim-1)
#define len(var) shape(var,0)
#define flen(var) fshape(var,0)
#define old_size(var) PyArray_SIZE((PyArrayObject *)(capi_ ## var ## _tmp))
/* #define index(i) capi_i ## i */
#define slen(var) capi_ ## var ## _len
#define size(var, ...) f2py_size((PyArrayObject *)(capi_ ## var ## _tmp), ## __VA_ARGS__, -1)

#ifdef DEBUGCFUNCS
#define CFUNCSMESS(mess) fprintf(stderr,"debug-capi:"mess);
#define CFUNCSMESSPY(mess,obj) CFUNCSMESS(mess) \
    PyObject_Print((PyObject *)obj,stderr,Py_PRINT_RAW);\
    fprintf(stderr,"\n");
#else
#define CFUNCSMESS(mess)
#define CFUNCSMESSPY(mess,obj)
#endif

#ifndef max
#define max(a,b) ((a > b) ? (a) : (b))
#endif
#ifndef min
#define min(a,b) ((a < b) ? (a) : (b))
#endif
#ifndef MAX
#define MAX(a,b) ((a > b) ? (a) : (b))
#endif
#ifndef MIN
#define MIN(a,b) ((a < b) ? (a) : (b))
#endif

#if defined(PREPEND_FORTRAN)
#if defined(NO_APPEND_FORTRAN)
#if defined(UPPERCASE_FORTRAN)
#define F_FUNC(f,F) _##F
#else
#define F_FUNC(f,F) _##f
#endif
#else
#if defined(UPPERCASE_FORTRAN)
#define F_FUNC(f,F) _##F##_
#else
#define F_FUNC(f,F) _##f##_
#endif
#endif
#else
#if defined(NO_APPEND_FORTRAN)
#if defined(UPPERCASE_FORTRAN)
#define F_FUNC(f,F) F
#else
#define F_FUNC(f,F) f
#endif
#else
#if defined(UPPERCASE_FORTRAN)
#define F_FUNC(f,F) F##_
#else
#define F_FUNC(f,F) f##_
#endif
#endif
#endif
#if defined(UNDERSCORE_G77)
#define F_FUNC_US(f,F) F_FUNC(f##_,F##_)
#else
#define F_FUNC_US(f,F) F_FUNC(f,F)
#endif


/************************ See f2py2e/cfuncs.py: cfuncs ************************/
static int f2py_size(PyArrayObject* var, ...)
{
  npy_int sz = 0;
  npy_int dim;
  npy_int rank;
  va_list argp;
  va_start(argp, var);
  dim = va_arg(argp, npy_int);
  if (dim==-1)
    {
      sz = PyArray_SIZE(var);
    }
  else
    {
      rank = PyArray_NDIM(var);
      if (dim>=1 && dim<=rank)
        sz = PyArray_DIM(var, dim-1);
      else
        fprintf(stderr, "f2py_size: 2nd argument value=%d fails to satisfy 1<=value<=%d. Result will be 0.\n", dim, rank);
    }
  va_end(argp);
  return sz;
}


/********************* See f2py2e/cfuncs.py: userincludes *********************/
/*need_userincludes*/

/********************* See f2py2e/capi_rules.py: usercode *********************/


/* See f2py2e/rules.py */
extern void reg_ev_energy(double*,double*,double*,double*,double*,double*,double*,double*,int,int,int,int);
extern void reg_ev_forces(double*,double*,double*,double*,double*,double*,double*,double*,double*,double*,int,int,int,int);
extern void reg_ev_all(double*,double*,double*,double*,double*,double*,double*,double*,double*,double*,double*,int,int,int,int);
/*eof externroutines*/

/******************** See f2py2e/capi_rules.py: usercode1 ********************/


/******************* See f2py2e/cb_rules.py: buildcallback *******************/
/*need_callbacks*/

/*********************** See f2py2e/rules.py: buildapi ***********************/

/******************************* reg_ev_energy *******************************/
static char doc_f2py_rout__tricube_reg_ev_energy[] = "\
val = reg_ev_energy(x0,x1,x2,f,fx0,fx1,fx2)\n\nWrapper for ``reg_ev_energy``.\
\n\nParameters\n----------\n"
"x0 : input rank-1 array('d') with bounds (ix)\n"
"x1 : input rank-1 array('d') with bounds (ix)\n"
"x2 : input rank-1 array('d') with bounds (ix)\n"
"f : input rank-3 array('d') with bounds (ix2,ix1,ix0)\n"
"fx0 : input rank-1 array('d') with bounds (ix0)\n"
"fx1 : input rank-1 array('d') with bounds (ix1)\n"
"fx2 : input rank-1 array('d') with bounds (ix2)\n"
"\nReturns\n-------\n"
"val : rank-1 array('d') with bounds (ix)";
/* extern void reg_ev_energy(double*,double*,double*,double*,double*,double*,double*,double*,int,int,int,int); */
static PyObject *f2py_rout__tricube_reg_ev_energy(const PyObject *capi_self,
                           PyObject *capi_args,
                           PyObject *capi_keywds,
                           void (*f2py_func)(double*,double*,double*,double*,double*,double*,double*,double*,int,int,int,int)) {
  PyObject * volatile capi_buildvalue = NULL;
  volatile int f2py_success = 1;
/*decl*/

  double *val = NULL;
  npy_intp val_Dims[1] = {-1};
  const int val_Rank = 1;
  PyArrayObject *capi_val_tmp = NULL;
  int capi_val_intent = 0;
  double *x0 = NULL;
  npy_intp x0_Dims[1] = {-1};
  const int x0_Rank = 1;
  PyArrayObject *capi_x0_tmp = NULL;
  int capi_x0_intent = 0;
  PyObject *x0_capi = Py_None;
  double *x1 = NULL;
  npy_intp x1_Dims[1] = {-1};
  const int x1_Rank = 1;
  PyArrayObject *capi_x1_tmp = NULL;
  int capi_x1_intent = 0;
  PyObject *x1_capi = Py_None;
  double *x2 = NULL;
  npy_intp x2_Dims[1] = {-1};
  const int x2_Rank = 1;
  PyArrayObject *capi_x2_tmp = NULL;
  int capi_x2_intent = 0;
  PyObject *x2_capi = Py_None;
  double *f = NULL;
  npy_intp f_Dims[3] = {-1, -1, -1};
  const int f_Rank = 3;
  PyArrayObject *capi_f_tmp = NULL;
  int capi_f_intent = 0;
  PyObject *f_capi = Py_None;
  double *fx0 = NULL;
  npy_intp fx0_Dims[1] = {-1};
  const int fx0_Rank = 1;
  PyArrayObject *capi_fx0_tmp = NULL;
  int capi_fx0_intent = 0;
  PyObject *fx0_capi = Py_None;
  double *fx1 = NULL;
  npy_intp fx1_Dims[1] = {-1};
  const int fx1_Rank = 1;
  PyArrayObject *capi_fx1_tmp = NULL;
  int capi_fx1_intent = 0;
  PyObject *fx1_capi = Py_None;
  double *fx2 = NULL;
  npy_intp fx2_Dims[1] = {-1};
  const int fx2_Rank = 1;
  PyArrayObject *capi_fx2_tmp = NULL;
  int capi_fx2_intent = 0;
  PyObject *fx2_capi = Py_None;
  int ix0 = 0;
  int ix1 = 0;
  int ix2 = 0;
  int ix = 0;
  static char *capi_kwlist[] = {"x0","x1","x2","f","fx0","fx1","fx2",NULL};

/*routdebugenter*/
#ifdef F2PY_REPORT_ATEXIT
f2py_start_clock();
#endif
  if (!PyArg_ParseTupleAndKeywords(capi_args,capi_keywds,\
    "OOOOOOO:_tricube.reg_ev_energy",\
    capi_kwlist,&x0_capi,&x1_capi,&x2_capi,&f_capi,&fx0_capi,&fx1_capi,&fx2_capi))
    return NULL;
/*frompyobj*/
  /* Processing variable x0 */
  ;
  capi_x0_intent |= F2PY_INTENT_IN|F2PY_INTENT_C;
  capi_x0_tmp = array_from_pyobj(NPY_DOUBLE,x0_Dims,x0_Rank,capi_x0_intent,x0_capi);
  if (capi_x0_tmp == NULL) {
    if (!PyErr_Occurred())
      PyErr_SetString(_tricube_error,"failed in converting 1st argument `x0' of _tricube.reg_ev_energy to C/Fortran array" );
  } else {
    x0 = (double *)(PyArray_DATA(capi_x0_tmp));

  /* Processing variable f */
  ;
  capi_f_intent |= F2PY_INTENT_IN|F2PY_INTENT_C;
  capi_f_tmp = array_from_pyobj(NPY_DOUBLE,f_Dims,f_Rank,capi_f_intent,f_capi);
  if (capi_f_tmp == NULL) {
    if (!PyErr_Occurred())
      PyErr_SetString(_tricube_error,"failed in converting 4th argument `f' of _tricube.reg_ev_energy to C/Fortran array" );
  } else {
    f = (double *)(PyArray_DATA(capi_f_tmp));

  /* Processing variable ix0 */
  ix0 = shape(f,2);
  /* Processing variable ix1 */
  ix1 = shape(f,1);
  /* Processing variable ix2 */
  ix2 = shape(f,0);
  /* Processing variable ix */
  ix = len(x0);
  /* Processing variable val */
  val_Dims[0]=ix;
  capi_val_intent |= F2PY_INTENT_OUT|F2PY_INTENT_HIDE|F2PY_INTENT_C;
  capi_val_tmp = array_from_pyobj(NPY_DOUBLE,val_Dims,val_Rank,capi_val_intent,Py_None);
  if (capi_val_tmp == NULL) {
    if (!PyErr_Occurred())
      PyErr_SetString(_tricube_error,"failed in converting hidden `val' of _tricube.reg_ev_energy to C/Fortran array" );
  } else {
    val = (double *)(PyArray_DATA(capi_val_tmp));

  /* Processing variable x1 */
  x1_Dims[0]=ix;
  capi_x1_intent |= F2PY_INTENT_IN|F2PY_INTENT_C;
  capi_x1_tmp = array_from_pyobj(NPY_DOUBLE,x1_Dims,x1_Rank,capi_x1_intent,x1_capi);
  if (capi_x1_tmp == NULL) {
    if (!PyErr_Occurred())
      PyErr_SetString(_tricube_error,"failed in converting 2nd argument `x1' of _tricube.reg_ev_energy to C/Fortran array" );
  } else {
    x1 = (double *)(PyArray_DATA(capi_x1_tmp));

  /* Processing variable x2 */
  x2_Dims[0]=ix;
  capi_x2_intent |= F2PY_INTENT_IN|F2PY_INTENT_C;
  capi_x2_tmp = array_from_pyobj(NPY_DOUBLE,x2_Dims,x2_Rank,capi_x2_intent,x2_capi);
  if (capi_x2_tmp == NULL) {
    if (!PyErr_Occurred())
      PyErr_SetString(_tricube_error,"failed in converting 3rd argument `x2' of _tricube.reg_ev_energy to C/Fortran array" );
  } else {
    x2 = (double *)(PyArray_DATA(capi_x2_tmp));

  /* Processing variable fx0 */
  fx0_Dims[0]=ix0;
  capi_fx0_intent |= F2PY_INTENT_IN|F2PY_INTENT_C;
  capi_fx0_tmp = array_from_pyobj(NPY_DOUBLE,fx0_Dims,fx0_Rank,capi_fx0_intent,fx0_capi);
  if (capi_fx0_tmp == NULL) {
    if (!PyErr_Occurred())
      PyErr_SetString(_tricube_error,"failed in converting 5th argument `fx0' of _tricube.reg_ev_energy to C/Fortran array" );
  } else {
    fx0 = (double *)(PyArray_DATA(capi_fx0_tmp));

  /* Processing variable fx1 */
  fx1_Dims[0]=ix1;
  capi_fx1_intent |= F2PY_INTENT_IN|F2PY_INTENT_C;
  capi_fx1_tmp = array_from_pyobj(NPY_DOUBLE,fx1_Dims,fx1_Rank,capi_fx1_intent,fx1_capi);
  if (capi_fx1_tmp == NULL) {
    if (!PyErr_Occurred())
      PyErr_SetString(_tricube_error,"failed in converting 6th argument `fx1' of _tricube.reg_ev_energy to C/Fortran array" );
  } else {
    fx1 = (double *)(PyArray_DATA(capi_fx1_tmp));

  /* Processing variable fx2 */
  fx2_Dims[0]=ix2;
  capi_fx2_intent |= F2PY_INTENT_IN|F2PY_INTENT_C;
  capi_fx2_tmp = array_from_pyobj(NPY_DOUBLE,fx2_Dims,fx2_Rank,capi_fx2_intent,fx2_capi);
  if (capi_fx2_tmp == NULL) {
    if (!PyErr_Occurred())
      PyErr_SetString(_tricube_error,"failed in converting 7th argument `fx2' of _tricube.reg_ev_energy to C/Fortran array" );
  } else {
    fx2 = (double *)(PyArray_DATA(capi_fx2_tmp));

/*end of frompyobj*/
#ifdef F2PY_REPORT_ATEXIT
f2py_start_call_clock();
#endif
/*callfortranroutine*/
        (*f2py_func)(val,x0,x1,x2,f,fx0,fx1,fx2,ix0,ix1,ix2,ix);
if (PyErr_Occurred())
  f2py_success = 0;
#ifdef F2PY_REPORT_ATEXIT
f2py_stop_call_clock();
#endif
/*end of callfortranroutine*/
    if (f2py_success) {
/*pyobjfrom*/
/*end of pyobjfrom*/
    CFUNCSMESS("Building return value.\n");
    capi_buildvalue = Py_BuildValue("N",capi_val_tmp);
/*closepyobjfrom*/
/*end of closepyobjfrom*/
    } /*if (f2py_success) after callfortranroutine*/
/*cleanupfrompyobj*/
  if((PyObject *)capi_fx2_tmp!=fx2_capi) {
    Py_XDECREF(capi_fx2_tmp); }
  }  /*if (capi_fx2_tmp == NULL) ... else of fx2*/
  /* End of cleaning variable fx2 */
  if((PyObject *)capi_fx1_tmp!=fx1_capi) {
    Py_XDECREF(capi_fx1_tmp); }
  }  /*if (capi_fx1_tmp == NULL) ... else of fx1*/
  /* End of cleaning variable fx1 */
  if((PyObject *)capi_fx0_tmp!=fx0_capi) {
    Py_XDECREF(capi_fx0_tmp); }
  }  /*if (capi_fx0_tmp == NULL) ... else of fx0*/
  /* End of cleaning variable fx0 */
  if((PyObject *)capi_x2_tmp!=x2_capi) {
    Py_XDECREF(capi_x2_tmp); }
  }  /*if (capi_x2_tmp == NULL) ... else of x2*/
  /* End of cleaning variable x2 */
  if((PyObject *)capi_x1_tmp!=x1_capi) {
    Py_XDECREF(capi_x1_tmp); }
  }  /*if (capi_x1_tmp == NULL) ... else of x1*/
  /* End of cleaning variable x1 */
  }  /*if (capi_val_tmp == NULL) ... else of val*/
  /* End of cleaning variable val */
  /* End of cleaning variable ix */
  /* End of cleaning variable ix2 */
  /* End of cleaning variable ix1 */
  /* End of cleaning variable ix0 */
  if((PyObject *)capi_f_tmp!=f_capi) {
    Py_XDECREF(capi_f_tmp); }
  }  /*if (capi_f_tmp == NULL) ... else of f*/
  /* End of cleaning variable f */
  if((PyObject *)capi_x0_tmp!=x0_capi) {
    Py_XDECREF(capi_x0_tmp); }
  }  /*if (capi_x0_tmp == NULL) ... else of x0*/
  /* End of cleaning variable x0 */
/*end of cleanupfrompyobj*/
  if (capi_buildvalue == NULL) {
/*routdebugfailure*/
  } else {
/*routdebugleave*/
  }
  CFUNCSMESS("Freeing memory.\n");
/*freemem*/
#ifdef F2PY_REPORT_ATEXIT
f2py_stop_clock();
#endif
  return capi_buildvalue;
}
/**************************** end of reg_ev_energy ****************************/

/******************************* reg_ev_forces *******************************/
static char doc_f2py_rout__tricube_reg_ev_forces[] = "\
val_dx0,val_dx1,val_dx2 = reg_ev_forces(x0,x1,x2,f,fx0,fx1,fx2)\n\nWrapper for ``reg_ev_forces``.\
\n\nParameters\n----------\n"
"x0 : input rank-1 array('d') with bounds (ix)\n"
"x1 : input rank-1 array('d') with bounds (ix)\n"
"x2 : input rank-1 array('d') with bounds (ix)\n"
"f : input rank-3 array('d') with bounds (ix2,ix1,ix0)\n"
"fx0 : input rank-1 array('d') with bounds (ix0)\n"
"fx1 : input rank-1 array('d') with bounds (ix1)\n"
"fx2 : input rank-1 array('d') with bounds (ix2)\n"
"\nReturns\n-------\n"
"val_dx0 : rank-1 array('d') with bounds (ix)\n"
"val_dx1 : rank-1 array('d') with bounds (ix)\n"
"val_dx2 : rank-1 array('d') with bounds (ix)";
/* extern void reg_ev_forces(double*,double*,double*,double*,double*,double*,double*,double*,double*,double*,int,int,int,int); */
static PyObject *f2py_rout__tricube_reg_ev_forces(const PyObject *capi_self,
                           PyObject *capi_args,
                           PyObject *capi_keywds,
                           void (*f2py_func)(double*,double*,double*,double*,double*,double*,double*,double*,double*,double*,int,int,int,int)) {
  PyObject * volatile capi_buildvalue = NULL;
  volatile int f2py_success = 1;
/*decl*/

  double *val_dx0 = NULL;
  npy_intp val_dx0_Dims[1] = {-1};
  const int val_dx0_Rank = 1;
  PyArrayObject *capi_val_dx0_tmp = NULL;
  int capi_val_dx0_intent = 0;
  double *val_dx1 = NULL;
  npy_intp val_dx1_Dims[1] = {-1};
  const int val_dx1_Rank = 1;
  PyArrayObject *capi_val_dx1_tmp = NULL;
  int capi_val_dx1_intent = 0;
  double *val_dx2 = NULL;
  npy_intp val_dx2_Dims[1] = {-1};
  const int val_dx2_Rank = 1;
  PyArrayObject *capi_val_dx2_tmp = NULL;
  int capi_val_dx2_intent = 0;
  double *x0 = NULL;
  npy_intp x0_Dims[1] = {-1};
  const int x0_Rank = 1;
  PyArrayObject *capi_x0_tmp = NULL;
  int capi_x0_intent = 0;
  PyObject *x0_capi = Py_None;
  double *x1 = NULL;
  npy_intp x1_Dims[1] = {-1};
  const int x1_Rank = 1;
  PyArrayObject *capi_x1_tmp = NULL;
  int capi_x1_intent = 0;
  PyObject *x1_capi = Py_None;
  double *x2 = NULL;
  npy_intp x2_Dims[1] = {-1};
  const int x2_Rank = 1;
  PyArrayObject *capi_x2_tmp = NULL;
  int capi_x2_intent = 0;
  PyObject *x2_capi = Py_None;
  double *f = NULL;
  npy_intp f_Dims[3] = {-1, -1, -1};
  const int f_Rank = 3;
  PyArrayObject *capi_f_tmp = NULL;
  int capi_f_intent = 0;
  PyObject *f_capi = Py_None;
  double *fx0 = NULL;
  npy_intp fx0_Dims[1] = {-1};
  const int fx0_Rank = 1;
  PyArrayObject *capi_fx0_tmp = NULL;
  int capi_fx0_intent = 0;
  PyObject *fx0_capi = Py_None;
  double *fx1 = NULL;
  npy_intp fx1_Dims[1] = {-1};
  const int fx1_Rank = 1;
  PyArrayObject *capi_fx1_tmp = NULL;
  int capi_fx1_intent = 0;
  PyObject *fx1_capi = Py_None;
  double *fx2 = NULL;
  npy_intp fx2_Dims[1] = {-1};
  const int fx2_Rank = 1;
  PyArrayObject *capi_fx2_tmp = NULL;
  int capi_fx2_intent = 0;
  PyObject *fx2_capi = Py_None;
  int ix0 = 0;
  int ix1 = 0;
  int ix2 = 0;
  int ix = 0;
  static char *capi_kwlist[] = {"x0","x1","x2","f","fx0","fx1","fx2",NULL};

/*routdebugenter*/
#ifdef F2PY_REPORT_ATEXIT
f2py_start_clock();
#endif
  if (!PyArg_ParseTupleAndKeywords(capi_args,capi_keywds,\
    "OOOOOOO:_tricube.reg_ev_forces",\
    capi_kwlist,&x0_capi,&x1_capi,&x2_capi,&f_capi,&fx0_capi,&fx1_capi,&fx2_capi))
    return NULL;
/*frompyobj*/
  /* Processing variable x0 */
  ;
  capi_x0_intent |= F2PY_INTENT_IN|F2PY_INTENT_C;
  capi_x0_tmp = array_from_pyobj(NPY_DOUBLE,x0_Dims,x0_Rank,capi_x0_intent,x0_capi);
  if (capi_x0_tmp == NULL) {
    if (!PyErr_Occurred())
      PyErr_SetString(_tricube_error,"failed in converting 1st argument `x0' of _tricube.reg_ev_forces to C/Fortran array" );
  } else {
    x0 = (double *)(PyArray_DATA(capi_x0_tmp));

  /* Processing variable f */
  ;
  capi_f_intent |= F2PY_INTENT_IN|F2PY_INTENT_C;
  capi_f_tmp = array_from_pyobj(NPY_DOUBLE,f_Dims,f_Rank,capi_f_intent,f_capi);
  if (capi_f_tmp == NULL) {
    if (!PyErr_Occurred())
      PyErr_SetString(_tricube_error,"failed in converting 4th argument `f' of _tricube.reg_ev_forces to C/Fortran array" );
  } else {
    f = (double *)(PyArray_DATA(capi_f_tmp));

  /* Processing variable ix0 */
  ix0 = shape(f,2);
  /* Processing variable ix1 */
  ix1 = shape(f,1);
  /* Processing variable ix2 */
  ix2 = shape(f,0);
  /* Processing variable ix */
  ix = len(x0);
  /* Processing variable val_dx0 */
  val_dx0_Dims[0]=ix;
  capi_val_dx0_intent |= F2PY_INTENT_OUT|F2PY_INTENT_HIDE|F2PY_INTENT_C;
  capi_val_dx0_tmp = array_from_pyobj(NPY_DOUBLE,val_dx0_Dims,val_dx0_Rank,capi_val_dx0_intent,Py_None);
  if (capi_val_dx0_tmp == NULL) {
    if (!PyErr_Occurred())
      PyErr_SetString(_tricube_error,"failed in converting hidden `val_dx0' of _tricube.reg_ev_forces to C/Fortran array" );
  } else {
    val_dx0 = (double *)(PyArray_DATA(capi_val_dx0_tmp));

  /* Processing variable val_dx1 */
  val_dx1_Dims[0]=ix;
  capi_val_dx1_intent |= F2PY_INTENT_OUT|F2PY_INTENT_HIDE|F2PY_INTENT_C;
  capi_val_dx1_tmp = array_from_pyobj(NPY_DOUBLE,val_dx1_Dims,val_dx1_Rank,capi_val_dx1_intent,Py_None);
  if (capi_val_dx1_tmp == NULL) {
    if (!PyErr_Occurred())
      PyErr_SetString(_tricube_error,"failed in converting hidden `val_dx1' of _tricube.reg_ev_forces to C/Fortran array" );
  } else {
    val_dx1 = (double *)(PyArray_DATA(capi_val_dx1_tmp));

  /* Processing variable val_dx2 */
  val_dx2_Dims[0]=ix;
  capi_val_dx2_intent |= F2PY_INTENT_OUT|F2PY_INTENT_HIDE|F2PY_INTENT_C;
  capi_val_dx2_tmp = array_from_pyobj(NPY_DOUBLE,val_dx2_Dims,val_dx2_Rank,capi_val_dx2_intent,Py_None);
  if (capi_val_dx2_tmp == NULL) {
    if (!PyErr_Occurred())
      PyErr_SetString(_tricube_error,"failed in converting hidden `val_dx2' of _tricube.reg_ev_forces to C/Fortran array" );
  } else {
    val_dx2 = (double *)(PyArray_DATA(capi_val_dx2_tmp));

  /* Processing variable x1 */
  x1_Dims[0]=ix;
  capi_x1_intent |= F2PY_INTENT_IN|F2PY_INTENT_C;
  capi_x1_tmp = array_from_pyobj(NPY_DOUBLE,x1_Dims,x1_Rank,capi_x1_intent,x1_capi);
  if (capi_x1_tmp == NULL) {
    if (!PyErr_Occurred())
      PyErr_SetString(_tricube_error,"failed in converting 2nd argument `x1' of _tricube.reg_ev_forces to C/Fortran array" );
  } else {
    x1 = (double *)(PyArray_DATA(capi_x1_tmp));

  /* Processing variable x2 */
  x2_Dims[0]=ix;
  capi_x2_intent |= F2PY_INTENT_IN|F2PY_INTENT_C;
  capi_x2_tmp = array_from_pyobj(NPY_DOUBLE,x2_Dims,x2_Rank,capi_x2_intent,x2_capi);
  if (capi_x2_tmp == NULL) {
    if (!PyErr_Occurred())
      PyErr_SetString(_tricube_error,"failed in converting 3rd argument `x2' of _tricube.reg_ev_forces to C/Fortran array" );
  } else {
    x2 = (double *)(PyArray_DATA(capi_x2_tmp));

  /* Processing variable fx0 */
  fx0_Dims[0]=ix0;
  capi_fx0_intent |= F2PY_INTENT_IN|F2PY_INTENT_C;
  capi_fx0_tmp = array_from_pyobj(NPY_DOUBLE,fx0_Dims,fx0_Rank,capi_fx0_intent,fx0_capi);
  if (capi_fx0_tmp == NULL) {
    if (!PyErr_Occurred())
      PyErr_SetString(_tricube_error,"failed in converting 5th argument `fx0' of _tricube.reg_ev_forces to C/Fortran array" );
  } else {
    fx0 = (double *)(PyArray_DATA(capi_fx0_tmp));

  /* Processing variable fx1 */
  fx1_Dims[0]=ix1;
  capi_fx1_intent |= F2PY_INTENT_IN|F2PY_INTENT_C;
  capi_fx1_tmp = array_from_pyobj(NPY_DOUBLE,fx1_Dims,fx1_Rank,capi_fx1_intent,fx1_capi);
  if (capi_fx1_tmp == NULL) {
    if (!PyErr_Occurred())
      PyErr_SetString(_tricube_error,"failed in converting 6th argument `fx1' of _tricube.reg_ev_forces to C/Fortran array" );
  } else {
    fx1 = (double *)(PyArray_DATA(capi_fx1_tmp));

  /* Processing variable fx2 */
  fx2_Dims[0]=ix2;
  capi_fx2_intent |= F2PY_INTENT_IN|F2PY_INTENT_C;
  capi_fx2_tmp = array_from_pyobj(NPY_DOUBLE,fx2_Dims,fx2_Rank,capi_fx2_intent,fx2_capi);
  if (capi_fx2_tmp == NULL) {
    if (!PyErr_Occurred())
      PyErr_SetString(_tricube_error,"failed in converting 7th argument `fx2' of _tricube.reg_ev_forces to C/Fortran array" );
  } else {
    fx2 = (double *)(PyArray_DATA(capi_fx2_tmp));

/*end of frompyobj*/
#ifdef F2PY_REPORT_ATEXIT
f2py_start_call_clock();
#endif
/*callfortranroutine*/
        (*f2py_func)(val_dx0,val_dx1,val_dx2,x0,x1,x2,f,fx0,fx1,fx2,ix0,ix1,ix2,ix);
if (PyErr_Occurred())
  f2py_success = 0;
#ifdef F2PY_REPORT_ATEXIT
f2py_stop_call_clock();
#endif
/*end of callfortranroutine*/
    if (f2py_success) {
/*pyobjfrom*/
/*end of pyobjfrom*/
    CFUNCSMESS("Building return value.\n");
    capi_buildvalue = Py_BuildValue("NNN",capi_val_dx0_tmp,capi_val_dx1_tmp,capi_val_dx2_tmp);
/*closepyobjfrom*/
/*end of closepyobjfrom*/
    } /*if (f2py_success) after callfortranroutine*/
/*cleanupfrompyobj*/
  if((PyObject *)capi_fx2_tmp!=fx2_capi) {
    Py_XDECREF(capi_fx2_tmp); }
  }  /*if (capi_fx2_tmp == NULL) ... else of fx2*/
  /* End of cleaning variable fx2 */
  if((PyObject *)capi_fx1_tmp!=fx1_capi) {
    Py_XDECREF(capi_fx1_tmp); }
  }  /*if (capi_fx1_tmp == NULL) ... else of fx1*/
  /* End of cleaning variable fx1 */
  if((PyObject *)capi_fx0_tmp!=fx0_capi) {
    Py_XDECREF(capi_fx0_tmp); }
  }  /*if (capi_fx0_tmp == NULL) ... else of fx0*/
  /* End of cleaning variable fx0 */
  if((PyObject *)capi_x2_tmp!=x2_capi) {
    Py_XDECREF(capi_x2_tmp); }
  }  /*if (capi_x2_tmp == NULL) ... else of x2*/
  /* End of cleaning variable x2 */
  if((PyObject *)capi_x1_tmp!=x1_capi) {
    Py_XDECREF(capi_x1_tmp); }
  }  /*if (capi_x1_tmp == NULL) ... else of x1*/
  /* End of cleaning variable x1 */
  }  /*if (capi_val_dx2_tmp == NULL) ... else of val_dx2*/
  /* End of cleaning variable val_dx2 */
  }  /*if (capi_val_dx1_tmp == NULL) ... else of val_dx1*/
  /* End of cleaning variable val_dx1 */
  }  /*if (capi_val_dx0_tmp == NULL) ... else of val_dx0*/
  /* End of cleaning variable val_dx0 */
  /* End of cleaning variable ix */
  /* End of cleaning variable ix2 */
  /* End of cleaning variable ix1 */
  /* End of cleaning variable ix0 */
  if((PyObject *)capi_f_tmp!=f_capi) {
    Py_XDECREF(capi_f_tmp); }
  }  /*if (capi_f_tmp == NULL) ... else of f*/
  /* End of cleaning variable f */
  if((PyObject *)capi_x0_tmp!=x0_capi) {
    Py_XDECREF(capi_x0_tmp); }
  }  /*if (capi_x0_tmp == NULL) ... else of x0*/
  /* End of cleaning variable x0 */
/*end of cleanupfrompyobj*/
  if (capi_buildvalue == NULL) {
/*routdebugfailure*/
  } else {
/*routdebugleave*/
  }
  CFUNCSMESS("Freeing memory.\n");
/*freemem*/
#ifdef F2PY_REPORT_ATEXIT
f2py_stop_clock();
#endif
  return capi_buildvalue;
}
/**************************** end of reg_ev_forces ****************************/

/********************************* reg_ev_all *********************************/
static char doc_f2py_rout__tricube_reg_ev_all[] = "\
val,val_dx0,val_dx1,val_dx2 = reg_ev_all(x0,x1,x2,f,fx0,fx1,fx2)\n\nWrapper for ``reg_ev_all``.\
\n\nParameters\n----------\n"
"x0 : input rank-1 array('d') with bounds (ix)\n"
"x1 : input rank-1 array('d') with bounds (ix)\n"
"x2 : input rank-1 array('d') with bounds (ix)\n"
"f : input rank-3 array('d') with bounds (ix2,ix1,ix0)\n"
"fx0 : input rank-1 array('d') with bounds (ix0)\n"
"fx1 : input rank-1 array('d') with bounds (ix1)\n"
"fx2 : input rank-1 array('d') with bounds (ix2)\n"
"\nReturns\n-------\n"
"val : rank-1 array('d') with bounds (ix)\n"
"val_dx0 : rank-1 array('d') with bounds (ix)\n"
"val_dx1 : rank-1 array('d') with bounds (ix)\n"
"val_dx2 : rank-1 array('d') with bounds (ix)";
/* extern void reg_ev_all(double*,double*,double*,double*,double*,double*,double*,double*,double*,double*,double*,int,int,int,int); */
static PyObject *f2py_rout__tricube_reg_ev_all(const PyObject *capi_self,
                           PyObject *capi_args,
                           PyObject *capi_keywds,
                           void (*f2py_func)(double*,double*,double*,double*,double*,double*,double*,double*,double*,double*,double*,int,int,int,int)) {
  PyObject * volatile capi_buildvalue = NULL;
  volatile int f2py_success = 1;
/*decl*/

  double *val = NULL;
  npy_intp val_Dims[1] = {-1};
  const int val_Rank = 1;
  PyArrayObject *capi_val_tmp = NULL;
  int capi_val_intent = 0;
  double *val_dx0 = NULL;
  npy_intp val_dx0_Dims[1] = {-1};
  const int val_dx0_Rank = 1;
  PyArrayObject *capi_val_dx0_tmp = NULL;
  int capi_val_dx0_intent = 0;
  double *val_dx1 = NULL;
  npy_intp val_dx1_Dims[1] = {-1};
  const int val_dx1_Rank = 1;
  PyArrayObject *capi_val_dx1_tmp = NULL;
  int capi_val_dx1_intent = 0;
  double *val_dx2 = NULL;
  npy_intp val_dx2_Dims[1] = {-1};
  const int val_dx2_Rank = 1;
  PyArrayObject *capi_val_dx2_tmp = NULL;
  int capi_val_dx2_intent = 0;
  double *x0 = NULL;
  npy_intp x0_Dims[1] = {-1};
  const int x0_Rank = 1;
  PyArrayObject *capi_x0_tmp = NULL;
  int capi_x0_intent = 0;
  PyObject *x0_capi = Py_None;
  double *x1 = NULL;
  npy_intp x1_Dims[1] = {-1};
  const int x1_Rank = 1;
  PyArrayObject *capi_x1_tmp = NULL;
  int capi_x1_intent = 0;
  PyObject *x1_capi = Py_None;
  double *x2 = NULL;
  npy_intp x2_Dims[1] = {-1};
  const int x2_Rank = 1;
  PyArrayObject *capi_x2_tmp = NULL;
  int capi_x2_intent = 0;
  PyObject *x2_capi = Py_None;
  double *f = NULL;
  npy_intp f_Dims[3] = {-1, -1, -1};
  const int f_Rank = 3;
  PyArrayObject *capi_f_tmp = NULL;
  int capi_f_intent = 0;
  PyObject *f_capi = Py_None;
  double *fx0 = NULL;
  npy_intp fx0_Dims[1] = {-1};
  const int fx0_Rank = 1;
  PyArrayObject *capi_fx0_tmp = NULL;
  int capi_fx0_intent = 0;
  PyObject *fx0_capi = Py_None;
  double *fx1 = NULL;
  npy_intp fx1_Dims[1] = {-1};
  const int fx1_Rank = 1;
  PyArrayObject *capi_fx1_tmp = NULL;
  int capi_fx1_intent = 0;
  PyObject *fx1_capi = Py_None;
  double *fx2 = NULL;
  npy_intp fx2_Dims[1] = {-1};
  const int fx2_Rank = 1;
  PyArrayObject *capi_fx2_tmp = NULL;
  int capi_fx2_intent = 0;
  PyObject *fx2_capi = Py_None;
  int ix0 = 0;
  int ix1 = 0;
  int ix2 = 0;
  int ix = 0;
  static char *capi_kwlist[] = {"x0","x1","x2","f","fx0","fx1","fx2",NULL};

/*routdebugenter*/
#ifdef F2PY_REPORT_ATEXIT
f2py_start_clock();
#endif
  if (!PyArg_ParseTupleAndKeywords(capi_args,capi_keywds,\
    "OOOOOOO:_tricube.reg_ev_all",\
    capi_kwlist,&x0_capi,&x1_capi,&x2_capi,&f_capi,&fx0_capi,&fx1_capi,&fx2_capi))
    return NULL;
/*frompyobj*/
  /* Processing variable x0 */
  ;
  capi_x0_intent |= F2PY_INTENT_IN|F2PY_INTENT_C;
  capi_x0_tmp = array_from_pyobj(NPY_DOUBLE,x0_Dims,x0_Rank,capi_x0_intent,x0_capi);
  if (capi_x0_tmp == NULL) {
    if (!PyErr_Occurred())
      PyErr_SetString(_tricube_error,"failed in converting 1st argument `x0' of _tricube.reg_ev_all to C/Fortran array" );
  } else {
    x0 = (double *)(PyArray_DATA(capi_x0_tmp));

  /* Processing variable f */
  ;
  capi_f_intent |= F2PY_INTENT_IN|F2PY_INTENT_C;
  capi_f_tmp = array_from_pyobj(NPY_DOUBLE,f_Dims,f_Rank,capi_f_intent,f_capi);
  if (capi_f_tmp == NULL) {
    if (!PyErr_Occurred())
      PyErr_SetString(_tricube_error,"failed in converting 4th argument `f' of _tricube.reg_ev_all to C/Fortran array" );
  } else {
    f = (double *)(PyArray_DATA(capi_f_tmp));

  /* Processing variable ix0 */
  ix0 = shape(f,2);
  /* Processing variable ix1 */
  ix1 = shape(f,1);
  /* Processing variable ix2 */
  ix2 = shape(f,0);
  /* Processing variable ix */
  ix = len(x0);
  /* Processing variable val */
  val_Dims[0]=ix;
  capi_val_intent |= F2PY_INTENT_OUT|F2PY_INTENT_HIDE|F2PY_INTENT_C;
  capi_val_tmp = array_from_pyobj(NPY_DOUBLE,val_Dims,val_Rank,capi_val_intent,Py_None);
  if (capi_val_tmp == NULL) {
    if (!PyErr_Occurred())
      PyErr_SetString(_tricube_error,"failed in converting hidden `val' of _tricube.reg_ev_all to C/Fortran array" );
  } else {
    val = (double *)(PyArray_DATA(capi_val_tmp));

  /* Processing variable val_dx0 */
  val_dx0_Dims[0]=ix;
  capi_val_dx0_intent |= F2PY_INTENT_OUT|F2PY_INTENT_HIDE|F2PY_INTENT_C;
  capi_val_dx0_tmp = array_from_pyobj(NPY_DOUBLE,val_dx0_Dims,val_dx0_Rank,capi_val_dx0_intent,Py_None);
  if (capi_val_dx0_tmp == NULL) {
    if (!PyErr_Occurred())
      PyErr_SetString(_tricube_error,"failed in converting hidden `val_dx0' of _tricube.reg_ev_all to C/Fortran array" );
  } else {
    val_dx0 = (double *)(PyArray_DATA(capi_val_dx0_tmp));

  /* Processing variable val_dx1 */
  val_dx1_Dims[0]=ix;
  capi_val_dx1_intent |= F2PY_INTENT_OUT|F2PY_INTENT_HIDE|F2PY_INTENT_C;
  capi_val_dx1_tmp = array_from_pyobj(NPY_DOUBLE,val_dx1_Dims,val_dx1_Rank,capi_val_dx1_intent,Py_None);
  if (capi_val_dx1_tmp == NULL) {
    if (!PyErr_Occurred())
      PyErr_SetString(_tricube_error,"failed in converting hidden `val_dx1' of _tricube.reg_ev_all to C/Fortran array" );
  } else {
    val_dx1 = (double *)(PyArray_DATA(capi_val_dx1_tmp));

  /* Processing variable val_dx2 */
  val_dx2_Dims[0]=ix;
  capi_val_dx2_intent |= F2PY_INTENT_OUT|F2PY_INTENT_HIDE|F2PY_INTENT_C;
  capi_val_dx2_tmp = array_from_pyobj(NPY_DOUBLE,val_dx2_Dims,val_dx2_Rank,capi_val_dx2_intent,Py_None);
  if (capi_val_dx2_tmp == NULL) {
    if (!PyErr_Occurred())
      PyErr_SetString(_tricube_error,"failed in converting hidden `val_dx2' of _tricube.reg_ev_all to C/Fortran array" );
  } else {
    val_dx2 = (double *)(PyArray_DATA(capi_val_dx2_tmp));

  /* Processing variable x1 */
  x1_Dims[0]=ix;
  capi_x1_intent |= F2PY_INTENT_IN|F2PY_INTENT_C;
  capi_x1_tmp = array_from_pyobj(NPY_DOUBLE,x1_Dims,x1_Rank,capi_x1_intent,x1_capi);
  if (capi_x1_tmp == NULL) {
    if (!PyErr_Occurred())
      PyErr_SetString(_tricube_error,"failed in converting 2nd argument `x1' of _tricube.reg_ev_all to C/Fortran array" );
  } else {
    x1 = (double *)(PyArray_DATA(capi_x1_tmp));

  /* Processing variable x2 */
  x2_Dims[0]=ix;
  capi_x2_intent |= F2PY_INTENT_IN|F2PY_INTENT_C;
  capi_x2_tmp = array_from_pyobj(NPY_DOUBLE,x2_Dims,x2_Rank,capi_x2_intent,x2_capi);
  if (capi_x2_tmp == NULL) {
    if (!PyErr_Occurred())
      PyErr_SetString(_tricube_error,"failed in converting 3rd argument `x2' of _tricube.reg_ev_all to C/Fortran array" );
  } else {
    x2 = (double *)(PyArray_DATA(capi_x2_tmp));

  /* Processing variable fx0 */
  fx0_Dims[0]=ix0;
  capi_fx0_intent |= F2PY_INTENT_IN|F2PY_INTENT_C;
  capi_fx0_tmp = array_from_pyobj(NPY_DOUBLE,fx0_Dims,fx0_Rank,capi_fx0_intent,fx0_capi);
  if (capi_fx0_tmp == NULL) {
    if (!PyErr_Occurred())
      PyErr_SetString(_tricube_error,"failed in converting 5th argument `fx0' of _tricube.reg_ev_all to C/Fortran array" );
  } else {
    fx0 = (double *)(PyArray_DATA(capi_fx0_tmp));

  /* Processing variable fx1 */
  fx1_Dims[0]=ix1;
  capi_fx1_intent |= F2PY_INTENT_IN|F2PY_INTENT_C;
  capi_fx1_tmp = array_from_pyobj(NPY_DOUBLE,fx1_Dims,fx1_Rank,capi_fx1_intent,fx1_capi);
  if (capi_fx1_tmp == NULL) {
    if (!PyErr_Occurred())
      PyErr_SetString(_tricube_error,"failed in converting 6th argument `fx1' of _tricube.reg_ev_all to C/Fortran array" );
  } else {
    fx1 = (double *)(PyArray_DATA(capi_fx1_tmp));

  /* Processing variable fx2 */
  fx2_Dims[0]=ix2;
  capi_fx2_intent |= F2PY_INTENT_IN|F2PY_INTENT_C;
  capi_fx2_tmp = array_from_pyobj(NPY_DOUBLE,fx2_Dims,fx2_Rank,capi_fx2_intent,fx2_capi);
  if (capi_fx2_tmp == NULL) {
    if (!PyErr_Occurred())
      PyErr_SetString(_tricube_error,"failed in converting 7th argument `fx2' of _tricube.reg_ev_all to C/Fortran array" );
  } else {
    fx2 = (double *)(PyArray_DATA(capi_fx2_tmp));

/*end of frompyobj*/
#ifdef F2PY_REPORT_ATEXIT
f2py_start_call_clock();
#endif
/*callfortranroutine*/
        (*f2py_func)(val,val_dx0,val_dx1,val_dx2,x0,x1,x2,f,fx0,fx1,fx2,ix0,ix1,ix2,ix);
if (PyErr_Occurred())
  f2py_success = 0;
#ifdef F2PY_REPORT_ATEXIT
f2py_stop_call_clock();
#endif
/*end of callfortranroutine*/
    if (f2py_success) {
/*pyobjfrom*/
/*end of pyobjfrom*/
    CFUNCSMESS("Building return value.\n");
    capi_buildvalue = Py_BuildValue("NNNN",capi_val_tmp,capi_val_dx0_tmp,capi_val_dx1_tmp,capi_val_dx2_tmp);
/*closepyobjfrom*/
/*end of closepyobjfrom*/
    } /*if (f2py_success) after callfortranroutine*/
/*cleanupfrompyobj*/
  if((PyObject *)capi_fx2_tmp!=fx2_capi) {
    Py_XDECREF(capi_fx2_tmp); }
  }  /*if (capi_fx2_tmp == NULL) ... else of fx2*/
  /* End of cleaning variable fx2 */
  if((PyObject *)capi_fx1_tmp!=fx1_capi) {
    Py_XDECREF(capi_fx1_tmp); }
  }  /*if (capi_fx1_tmp == NULL) ... else of fx1*/
  /* End of cleaning variable fx1 */
  if((PyObject *)capi_fx0_tmp!=fx0_capi) {
    Py_XDECREF(capi_fx0_tmp); }
  }  /*if (capi_fx0_tmp == NULL) ... else of fx0*/
  /* End of cleaning variable fx0 */
  if((PyObject *)capi_x2_tmp!=x2_capi) {
    Py_XDECREF(capi_x2_tmp); }
  }  /*if (capi_x2_tmp == NULL) ... else of x2*/
  /* End of cleaning variable x2 */
  if((PyObject *)capi_x1_tmp!=x1_capi) {
    Py_XDECREF(capi_x1_tmp); }
  }  /*if (capi_x1_tmp == NULL) ... else of x1*/
  /* End of cleaning variable x1 */
  }  /*if (capi_val_dx2_tmp == NULL) ... else of val_dx2*/
  /* End of cleaning variable val_dx2 */
  }  /*if (capi_val_dx1_tmp == NULL) ... else of val_dx1*/
  /* End of cleaning variable val_dx1 */
  }  /*if (capi_val_dx0_tmp == NULL) ... else of val_dx0*/
  /* End of cleaning variable val_dx0 */
  }  /*if (capi_val_tmp == NULL) ... else of val*/
  /* End of cleaning variable val */
  /* End of cleaning variable ix */
  /* End of cleaning variable ix2 */
  /* End of cleaning variable ix1 */
  /* End of cleaning variable ix0 */
  if((PyObject *)capi_f_tmp!=f_capi) {
    Py_XDECREF(capi_f_tmp); }
  }  /*if (capi_f_tmp == NULL) ... else of f*/
  /* End of cleaning variable f */
  if((PyObject *)capi_x0_tmp!=x0_capi) {
    Py_XDECREF(capi_x0_tmp); }
  }  /*if (capi_x0_tmp == NULL) ... else of x0*/
  /* End of cleaning variable x0 */
/*end of cleanupfrompyobj*/
  if (capi_buildvalue == NULL) {
/*routdebugfailure*/
  } else {
/*routdebugleave*/
  }
  CFUNCSMESS("Freeing memory.\n");
/*freemem*/
#ifdef F2PY_REPORT_ATEXIT
f2py_stop_clock();
#endif
  return capi_buildvalue;
}
/***************************** end of reg_ev_all *****************************/
/*eof body*/

/******************* See f2py2e/f90mod_rules.py: buildhooks *******************/
/*need_f90modhooks*/

/************** See f2py2e/rules.py: module_rules['modulebody'] **************/

/******************* See f2py2e/common_rules.py: buildhooks *******************/

/*need_commonhooks*/

/**************************** See f2py2e/rules.py ****************************/

static FortranDataDef f2py_routine_defs[] = {
  {"reg_ev_energy",-1,{{-1}},0,(char *)reg_ev_energy,(f2py_init_func)f2py_rout__tricube_reg_ev_energy,doc_f2py_rout__tricube_reg_ev_energy},
  {"reg_ev_forces",-1,{{-1}},0,(char *)reg_ev_forces,(f2py_init_func)f2py_rout__tricube_reg_ev_forces,doc_f2py_rout__tricube_reg_ev_forces},
  {"reg_ev_all",-1,{{-1}},0,(char *)reg_ev_all,(f2py_init_func)f2py_rout__tricube_reg_ev_all,doc_f2py_rout__tricube_reg_ev_all},

/*eof routine_defs*/
  {NULL}
};

static PyMethodDef f2py_module_methods[] = {

  {NULL,NULL}
};

#if PY_VERSION_HEX >= 0x03000000
static struct PyModuleDef moduledef = {
  PyModuleDef_HEAD_INIT,
  "_tricube",
  NULL,
  -1,
  f2py_module_methods,
  NULL,
  NULL,
  NULL,
  NULL
};
#endif

#if PY_VERSION_HEX >= 0x03000000
#define RETVAL m
PyMODINIT_FUNC PyInit__tricube(void) {
#else
#define RETVAL
PyMODINIT_FUNC init_tricube(void) {
#endif
  int i;
  PyObject *m,*d, *s;
#if PY_VERSION_HEX >= 0x03000000
  m = _tricube_module = PyModule_Create(&moduledef);
#else
  m = _tricube_module = Py_InitModule("_tricube", f2py_module_methods);
#endif
  Py_TYPE(&PyFortran_Type) = &PyType_Type;
  import_array();
  if (PyErr_Occurred())
    {PyErr_SetString(PyExc_ImportError, "can't initialize module _tricube (failed to import numpy)"); return RETVAL;}
  d = PyModule_GetDict(m);
  s = PyString_FromString("$Revision: $");
  PyDict_SetItemString(d, "__version__", s);
#if PY_VERSION_HEX >= 0x03000000
  s = PyUnicode_FromString(
#else
  s = PyString_FromString(
#endif
    "This module '_tricube' is auto-generated with f2py (version:2).\nFunctions:\n"
"  val = reg_ev_energy(x0,x1,x2,f,fx0,fx1,fx2)\n"
"  val_dx0,val_dx1,val_dx2 = reg_ev_forces(x0,x1,x2,f,fx0,fx1,fx2)\n"
"  val,val_dx0,val_dx1,val_dx2 = reg_ev_all(x0,x1,x2,f,fx0,fx1,fx2)\n"
".");
  PyDict_SetItemString(d, "__doc__", s);
  _tricube_error = PyErr_NewException ("_tricube.error", NULL, NULL);
  Py_DECREF(s);
  for(i=0;f2py_routine_defs[i].name!=NULL;i++)
    PyDict_SetItemString(d, f2py_routine_defs[i].name,PyFortranObject_NewAsAttr(&f2py_routine_defs[i]));



/*eof initf2pywraphooks*/
/*eof initf90modhooks*/

/*eof initcommonhooks*/


#ifdef F2PY_REPORT_ATEXIT
  if (! PyErr_Occurred())
    on_exit(f2py_report_on_exit,(void*)"_tricube");
#endif

  return RETVAL;
}
#ifdef __cplusplus
}
#endif
