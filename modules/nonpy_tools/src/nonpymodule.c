

#include <Python.h>
#include <stdio.h>

#include "signature.h"

/* Use this macro to set up the module name. Do not use quotes.
   This name will be used in various places like the init function
   name as well as to generate the string to be placed in the Python
   module
*/

#define THIS_MODULE_NAME _nonpy_tools


/* Some misc macros */
#define _CONCAT(a,b) a ## b
#define CONCAT(a,b) _CONCAT(a,b)
#define _STR(a) # a
#define STR(a) _STR(a)

#if defined(__GNUC__)
#  define UNUSED_VAR(x) CONCAT(UNUSED_, x) __attribute__((unused))
#elif defined(__LCLINT__)
#  define UNUSED_VAR(x) /*@unused@*/ CONCAT(UNUSED_, x)
#elif defined(__cplusplus)
#  define UNUSED_VAR(x)
#else
#  define UNUSED_VAR(x) CONCAT(UNUSED_, x)
#endif 

/* Python 3 support */
#if PY_MAJOR_VERSION >= 3
#   define PYTHON3
#   define MOD_INIT(name) PyMODINIT_FUNC CONCAT(PyInit_, name)(void)
#   define MOD_RETURN(val) do { return val; } while(0)
#else
#   define MOD_INIT(name) PyMODINIT_FUNC CONCAT(init, name)(void)
#   define MOD_RETURN(val) do {} while(0)
#endif

/* Box a signature in a Python structure.
   The structure used is a tuple containing:
   - a tuple with nin, nout, nargs
   - an integer with the number of dimension variables
   - a tuple with the tuples for each argument and their bindings to the
     dimension variables
*/
static PyObject *
box_signature(parsed_signature *ps)
{
    PyObject *rv = PyTuple_New(3);

    if (rv) {
        /* from this point, checking results could be improved */
        PyTuple_SET_ITEM(rv, 0, Py_BuildValue("(nnn)", 
                                              ps->input_count,
                                              ps->output_count,
                                              ps->arg_count));
        PyTuple_SET_ITEM(rv, 1, PyLong_FromSize_t(ps->dimension_variable_count));
        {
            size_t arg_count = ps->arg_count;
            PyObject *arg_dim_tuple = PyTuple_New(arg_count);

            for (size_t arg = 0; arg < arg_count; arg++) {
                size_t dim_count = ps->arg_dimension_count[arg];
                PyObject *dim_tuple = PyTuple_New(dim_count);
                size_t *dim_data = ps->arg_shape_idx +
                    ps->arg_shape_offsets[arg];
                for (size_t dim = 0; dim < dim_count; dim++) {
                    PyTuple_SET_ITEM(dim_tuple, dim,
                                     PyLong_FromSize_t(dim_data[dim]));
                }
                PyTuple_SET_ITEM(arg_dim_tuple, arg, dim_tuple);
            }

            PyTuple_SET_ITEM(rv, 2, arg_dim_tuple);
        }
    }

    return rv;
}

/* return the data from the signature boxed in some Python structure.
*/
static PyObject *
legacy_parse_signature(PyObject *UNUSED_VAR(self),
                       PyObject *args,
                       PyObject *UNUSED_VAR(kwargs))
{
    parsed_signature *ps;
    long int nin, nargs;
    char *signature;
    if (!PyArg_ParseTuple(args, "sll", &signature, &nin, &nargs))
        return NULL;

    ps = legacy_numpy_parse_signature(signature, nin, nargs);

    if (ps) {
        return box_signature(ps);
    } else {
        return PyErr_Format(PyExc_RuntimeError, "Parse error on signature '%s'", signature);
    }
}

/* The method table */
static struct PyMethodDef methods[] = {
    { "legacy_parse_signature",
      legacy_parse_signature,
      METH_VARARGS, NULL },
    { NULL, NULL, 0, NULL }   /* sentinel */
};

#if defined(PYTHON3)
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    STR(THIS_MODULE_NAME),
    NULL,
    -1,
    methods,
    NULL,
    NULL,
    NULL,
    NULL
};
#endif 

MOD_INIT(_nonpy_tools)
{
    PyObject *m = NULL;

#if defined(PYTHON3)
    m = PyModule_Create(&moduledef);
#else
    m = Py_InitModule(STR(THIS_MODULE_NAME), methods);
#endif /* PYTHON3 */

    MOD_RETURN(m);
}
