

#include "Python.h"

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

static PyObject *
test_func(PyObject *UNUSED_VAR(self),
          PyObject *UNUSED_VAR(args),
          PyObject *UNUSED_VAR(kwargs))
{
    return PyUnicode_FromString("Hello World!");
}

/* The method table */
static struct PyMethodDef methods[] = {
    { "test_func",
      (PyCFunction) test_func,
      METH_NOARGS, NULL },
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
