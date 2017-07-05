=====================================
 NumPy's gufunc implementation guide
=====================================

This document presents the current gufunc implementation in NumPy. The
idea being performing a dissection of it, pointing important parts of
the code, its limitations and how it could be improved.

The objective would be to be able to mimic its current behavior and,
eventually, improve it. All this while removing as many dependencies
with NumPy as possible, so that a simple layer (adapter) could be used
to run it over NumPy arrays. A similar adapter could be used for
different kinds of arrays, including a mixture of them.

Note that the NumPy code dealing with (g)ufuncs are in
'numpy/core/src/umath'.


Python structure of a gufunc
============================

In NumPy, ufuncs and gufuncs are objects. Both usethe same underlying
C module object, configured slightly differently (gufuncs were
implemented on top of ufuncs, as an extension). Both will return
"numpy.ufunc" as their "type".

Basic operations on arrays are implemented as ufuncs, while more
complex operations require gufuncs. 

.. code:: python
   import numpy as np
   from numpy.linalg import _umath_linalg as _ula

   type(np.add)

   >>> numpy.ufunc

   type(_ula.eig)

   >>> numpy.ufunc



Basic operations on arrays in NumPy are ufuncs (addition,
substraction, etc...), while there are some other 

In NumPy a gufunc is an object. There are several examples of gufuncs
in a private module in numpy that implements gufuncs for linear algebra.
(numpy.linalg._umath_linalg).

From the python environment, some information is accessible that helps
identify characteristics in a gufunc. As an example we will take the
eig gufunc.

.. code:: python
   import numpy.linalg._umath_linalg as _ula


Resolving
=========

Signature
---------

The first important data to NumPy's gufuncs is the signature. The
signature specifies the inner shape of the parameters and
results. That is, the shape of the arguments that a single call to the
kernel would expect.

.. code:: python
   _ula.eig.signature

   >>> '(m,m)->(m),(m,m)'

Note that the signature appears as a string. The used letters
determine different variables when matching. So if the same letter is
used, the dimensions that letter appears in MUST match.

The elements to the left of the '->' are the input parameters
shape. In this case, '(m,m)' means that the kernel requires the first
parameter to be a square matrix, as both dimensions use the same
letters. This will match (3,3), (4,4) but not (3,4), for example. If
the specified shape was '(m,n)', any two-dimensional matrix will match.

The elements to the right of the '->' are the output result shapes. In
this case there are two output parameters, the first being
one-dimensional of shape (m) and the second bi-dimensional of shape
(m,m). Note that the actual values of m for a given gufunc call will
be inferred based on the value of m found in the input parameters.


Limitations
~~~~~~~~~~~

- It is not possible to specify a dimension with a literal. That is,
  it is not possible to specify something like (n,3)->(4).

- It is not possible to specify a dimension for an output with a
  length not present in any of the inputs like (n, n)->(m).

- It is not possible to specify a dimension in the output with a
  expression based on the inputs like (m,n)->(min(m,n)).

There are trivial examples where functions which could make
interesting gufuncs can't be handled due to the limitations in the
signature specification:

- A function to convert a quaternion to an equivalent rotation matrix:
  It would require a (4)->(3,3) signature. Not only it is impossible
  to specify with literals 4 and 3, but also it is not possible to use
  (m)->(n,n) that would allow a more general match that will include
  the interesting case as "3" does not appear as a dimension in the
  input parameter.

- Cases like SVD where, when full_matrices is False, some output
  dimensions depend on the minimum of two input dimensions:
  (M,N)->(M,K),(K,N) where K = min(M,N). This is worked around in
  NumPy by having different gufuncs for the different cases (all
  wrapped by a common Python function that selects the actual gufunc
  to use).


Loops
-----

Even as the (g)ufunc shows as a single function in Python, it is
implemented as a series of specialized loops that are selected by
type. From Python, it is possible to check the loop types in a gufunc
by inspecting the "types" instance variable of a gufunc. The instance
variable holds a list with an element for each supported loop. Each
loop has associated to a C function that implements the logic needed
for that particular type combination. 

Note that these loops in NumPy source code are implemented using some
preprocessing to avoid repeating code where only types difer.

.. code:: python
   _ula.eig.types

   >>> ['f->FF', 'd->DD', 'D->DD']

The types are shown using NumPy's single letter types. One letter for
each input and output with '->' as a separator between inputs and
outputs.

Datashapes for loops in NumPy's gufuncs are thus split into the
signature and types. Signature is shared for all loops, while the
underlying basic types are loop specific.


Loop selection
~~~~~~~~~~~~~~

In NumPy's gufunc, the signature (shared for all loops in a gufunc)
must be able to match the shapes of the input parameters. From that
information the machinery can know what forms the inner shape of the
inputs and, due to the restrictions in NumPy's signature
specification, it is enough to resolve the inner shape of the outputs.

The outer shape of the inputs will conform the actual "shape" of
execution. This may involve broadcasting rules. That outer shape
will be the outer shape of the outputs.

The loop to use is selected based on the input types and the available
type strings. This is typically done by going down the list checking
the input dtypes against those in the loop applying NumPy's coercion
rules. As a match is found, the associated loop is selected. If no
match is found, an error is raised. So ordering of the loops is
important. NOTE: This is the behavior of the default type
resolver. NumPy implements other type resolvers for specific
cases. This hints that type resolving is something that may not have a
generic satisfactory solution.

After the loop outer shape is inferred from the input shapes and the
execution loop is selected (and thus, output types become known),
there is enough information to allocate the output arrays.


Execution Preparation
=====================

Execution
=========


Execution wind-down
===================
