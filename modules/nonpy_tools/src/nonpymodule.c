

#include <Python.h>
#include <structmember.h>

#include <stdio.h>
#include <stdint.h>

#include "signature.h"

/* Use this macro to set up the module name. Do not use quotes.
   This name will be used in various places like the init function
   name as well as to generate the string to be placed in the Python
   module
*/

#define THIS_MODULE_PATH "gufunctools"
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

/* -----------------------------------------------------------------------------
 * Boxed signature object
 */

typedef struct {
    PyObject_HEAD
    parsed_signature *the_signature;
} guft_SignatureObject;

static void
Signature_dealloc(guft_SignatureObject *self)
{
    /* the wrapper is owner of the underlying object */
    release_parsed_signature(self->the_signature);
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject *
Signature_new(PyTypeObject *type,
              PyObject *args,
              PyObject *kwds)
{
    guft_SignatureObject *self;

    self = (guft_SignatureObject *) type->tp_alloc(type, 0);
    if (self != NULL) {
        self->the_signature = NULL;
    }

    return (PyObject *)self;
}

static int
Signature_init(guft_SignatureObject *self,
               PyObject *args,
               PyObject *kwds)
{
    const char *signature_str = NULL;

    static char *kwlist[] = { "id",  NULL };

    if (! PyArg_ParseTupleAndKeywords(args, kwds, "s|", kwlist,
                                      &signature_str))
        return -1;

    if (signature_str) {
        self->the_signature = numpy_parse_signature(signature_str);
    }

    return 0;
}

static PyObject *
Signature_boxed(guft_SignatureObject *self)
{
    return box_signature(self->the_signature);
}

#if SIZEOF_UINTPTR_T == SIZEOF_LONG
#   define T_UINTPTR T_ULONG
#elif SIZEOF_UINTPTR_T == SIZEOF_LONG_LONG
#   define T_UINTPTR T_ULONGLONG
#else
#   error "Don't know what type to use to expose naked pointers in Python."
#endif
static PyMemberDef guft_SignatureObject_members[] = {
    {"id", T_UINTPTR, offsetof(guft_SignatureObject, the_signature), READONLY, 
     "c object ptr"},
    {NULL} /* Sentinel */
};

static PyMethodDef guft_SignatureObject_methods[] = {
    {"boxed", (PyCFunction)Signature_boxed, METH_NOARGS,
     "Returns signature data boxed in python tuples"
    },
    {NULL} /* Sentinel */
};


static PyTypeObject guft_SignatureType = {
    PyVarObject_HEAD_INIT(NULL,0)
    THIS_MODULE_PATH"."STR(THIS_MODULE_NAME)".Signature", /* tp_name */
    sizeof(guft_SignatureObject),                         /* tp_basicsize */
    0,                                                    /* tp_itemsize */
    (destructor)Signature_dealloc,                        /* tp_dealloc */
    0,                                                    /* tp_print */
    0,                                                    /* tp_getattr */
    0,                                                    /* tp_setattr */
    0,                                                    /* tp_reserved */
    0,                                                    /* tp_repr */
    0,                                                    /* tp_as_number */
    0,                                                    /* tp_as_sequence */
    0,                                                    /* tp_as_mapping */
    0,                                                    /* tp_hash */
    0,                                                    /* tp_call */
    0,                                                    /* tp_str */
    0,                                                    /* tp_getattro */
    0,                                                    /* tp_setattro */
    0,                                                    /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,             /* tp_flags */
    "Dimension signature objects",                        /* tp_doc */
    0,                                                    /* tp_traverse */
    0,                                                    /* tp_clear */
    0,                                                    /* tp_richcompare */
    0,                                                    /* tp_weaklistoffset */
    0,                                                    /* tp_iter */
    0,                                                    /* tp_iternext */
    guft_SignatureObject_methods,                         /* tp_methods */
    guft_SignatureObject_members,                         /* tp_members */
    0,                                                    /* tp_getset */
    0,                                                    /* tp_base */
    0,                                                    /* tp_dict */
    0,                                                    /* tp_descr_get */
    0,                                                    /* tp_descr_set */
    0,                                                    /* tp_dictoffset */
    (initproc)Signature_init,                             /* tp_init */
    0,                                                    /* tp_alloc */
    Signature_new,                                        /* tp_new */
};


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

static PyObject *
parse_signature(PyObject *UNUSED_VAR(self),
                       PyObject *args,
                       PyObject *UNUSED_VAR(kwargs))
{
    parsed_signature *ps;
    char *signature;
    if (!PyArg_ParseTuple(args, "s", &signature))
        return NULL;

    ps = numpy_parse_signature(signature);

    if (ps) {
        return box_signature(ps);
    } else {
        return PyErr_Format(PyExc_RuntimeError, "Parse error on signature '%s'", signature);
    }
}

/* The method table */
static struct PyMethodDef methods[] = {
    { "legacy_parse_signature",
      (PyCFunction)legacy_parse_signature,
      METH_VARARGS, NULL },
    { "parse_signature",
      (PyCFunction)parse_signature,
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

    guft_SignatureType.tp_new = PyType_GenericNew;
    if (PyType_Ready(&guft_SignatureType) < 0)
        MOD_RETURN(m);

#if defined(PYTHON3)
    m = PyModule_Create(&moduledef);
#else
    m = Py_InitModule(STR(THIS_MODULE_NAME), methods);
#endif /* PYTHON3 */

    Py_INCREF(&guft_SignatureType);

    PyModule_AddObject(m, "Signature", (PyObject*) &guft_SignatureType);

    MOD_RETURN(m);
}
