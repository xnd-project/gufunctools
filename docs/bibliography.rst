================================================
 A small bibliography of gufunc like  functions
================================================

What is a gufunc like function?
===============================

gufunc stands for "**g**\ eneralized **u**\ niversal **func**\
tion". The idea being a usually small function that is applied to an
array of input arguments to produce an array of results.

NumPy
=====

We take the concept and name from NumPy (surely some similar concepts
exist in other libraries/languages).

In NumPy there are two different revisions of the concept:

The original ufuncs (**u**\ niversal **func**\ tion):

  A vectorized function based on scalar arguments. Used to implement
  the basic operations for arrays.

gufuncs (**g**\ eneralized **u**\ niversal **func**\ tion):

  A generalization of the ufuncs where the arguments are not limited
  to scalars, but can "consume" inner dimensions.

In NumPy arrays have a base type and a set of dimensions (the
shape). In order to be able to apply a ufunc over a set of arguments
the shapes of the arguments must match. In order to match the shapes
must be compatible. Shapes are compatible if they the shapes are
exactly equal or if one can be promoted to another by *broadcasting*.
*Broadcasting* adds dimensions to any required size by logically
replicating the inner dimensions. For example, an array of shape (3,)
could be *broadcast* into an array of shape (n, 3) that logically has
the same values in array[i, :] for all i. Those values matching the
elements of the *broadcasted* array.

In NumPy the broadcast happens logically, actual copies are not
performed.

TODO: Document all the behavior of gufuncs in NumPy, including the
implications that fancy indensing, masking and all the indexing
features of NumPy may have on ufunc execution. With a richer array
meta-data some extra operations could be performed by the gufunc
machinery by default

 - only array arguments in gufuncs. It would be nice to be able to
   provide "uniform" parameters, that is parameters that are constant
   for all the array but could change per call. While it would be
   possible to rely on broadcasting for those, certain back-ends may
   benefit of the constraint -for example, passing it in constants
   memory in GPUs-.

 - signature limitations for gufuncs


vectorize and frompyfunc
------------------------

NumPy provides a vectorize call that allows to run a single Python
function over the array.

Even if they produce ufuncs, the resulting ufuncs are quite slow which
limits its practical usage.


Numba
=====

Numba provides means to build NumPy ufuncs and gufuncs in a way that
is performing. There are versions to generate CPU and GPU implementations.

The CPU implementations rely quite a bit on the NumPy machinery (as
far as I know). The GPU backend uses a different strategy.

I have to investigate this a bit more.


Apache ARROW
============

Columnar In-Memory Analytics. Idea of a common data layer with a wire
format. Different tools understanding natively that data layer could
avoid conversion. Arrow also claims to use zero-copy memory sharing
IPC, allowing arrays to be shared between processes with no actual
memory copying.

.. [ARROW] https://arrow.apache.org/


DESOLA
======

This one was provided by Stuart and is worth investigating. It seems
to describe a system building "full solutions" from a composition of
functions in a way that may be desirable for our gufuncs in the long
term.

.. [DESOLA] http://www.doc.ic.ac.uk/~fpr02/phd_thesis.pdf 


APL
===

APL sure has some stuff we could take inspiration from.
