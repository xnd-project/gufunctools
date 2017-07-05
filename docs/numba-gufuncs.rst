=====================================
 Numba's gufunc implementation guide
=====================================

This document presents the ufunc/gufunc implementation in numba. Numba
supports the generation of ufuncs/gufuncs via the numba.vectorize and
numba.guvectorize decorators.

Numba contains several different implementations of gufuncs. The code
for the CPU targets relies heavily on NumPy's gufunc code. Numba also
supports "DUFuncs", that are just gufuncs where the table of kernels
is dynamic. A DUFunc is using (and modifying at runtime) a NumPy
gufunc.

There is also an implementations for 'gpu' (cuda) that does not depend
on NumPy's gufunc machinery (it depends on cuda machinery instead).

Some info can be seen in
