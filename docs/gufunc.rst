================================
 Towards a better gufunc system
================================

In Plures we are looking to improve the ideas of NumPy's gufuncs.

The objective being building an independent library able to provide
gufunc-like functionality with a minimal set of dependencies, and
being able to work on different data containers.


What is a gufunc
================

A gufunc is a function that is intended to work on arrays of operands.
The idea is to execute the same function (kernel) on different arguments.
The arguments and the results will be organized as arrays.

There are several important concepts related to a gufunc:

- Gufunc: The object that describes a function that operates over a
  vector of arguments. This is the object that will be called with
  inputs to generate outputs by applying the kernel as many times as
  needed.

- Kernel: Operations to be performed on the arguments. The program to
  execute for each of the vector inputs.

- Signature: It is the description of the parameters as seen by the
  kernel. This describes both, the shape and type of the parameters
  required by the kernel.

- Inputs: Parameters in array form that will be fed to the
  Kernel. Typically, one entry of each input will be passed per Kernel
  call.

- Outputs: These are the ouputs that the Kernel will spit out. Each
  call to the kernel will result in a single output entry for each of
  the outputs.

- Uniform (input) parameters: These are parameters that remain constant for
  all kernel calls for a given gufunc invocation. They may vary from
  invocation to invocation. NumPy's gufunc do not support these (although
  equivalent behavior can be had relying on broadcasting).

- Inner shape: The shape of an Input (Output) that is considered an
  element. That is, the shape that parameter received by the kernel
  will have.

- Outer shape: The remaining of an Input (Output) shape that is not
  the Inner shape. The outer shapes define how many times a Kernel
  will be called.

- Element: An element defines the parameters for a single kernel call.
  The total number of elements will be defined by the outer shapes of
  the inputs.

- Datashape (abstract): A composition of dimensions and base type. This
  describes the "abstract" portion of the datashape without any layout
  details. It would look something like "9 * 7 * float32". Abstract
  datashape should be enough to reason about operations, but will not
  contain enough information to perform them.

- Datashape (concrete): Abstract datashape plus metadata describing a
  memory layout. A concrete datashape plus a memory pointer has enough
  information to perform operations on the data. It specifies the
  actual layout in memory. Its associated meta-data should be enough
  to transform logical indices into memory addresses relative to the
  "data pointer".

- Data pointer: A base value (or set of values) that allow accessing
  the data of a given array. Combined with a concrete datashape, it
  forms an array view that can be inspected and operated on.


Critical Pieces
===============

The gufunc has several key pieces:

- Resolving/dispatching: taking the arguments and based on the gufunc
  description, the arguments must be checked and resolved. This includes:

  - Resolving signature variables (usually dimension matching arguments).

  - Resolving involved types (based on the arguments)

  - Kernel selection (based on types)

  - Infer result dimensions/types (based on signature variables,
    involved types and kernel specific information)

  - Identify outer shape. At this point the number of calls to the
    kernel should already be evident. Note that this is quite related
    to the result dimensions.

- Execution preparation. This may involve allocating temporary buffers
  and other resources needed for execution, including setting up
  anything related to exception handling.

- Execution. This should make the appropriate calls to the kernel. The
  kernel may be arranged so that execution can be performed in
  bulk. That is, the kernel function will typically be able to handle
  more than one kernel in one call. Kernel should be able to handle
  "per element" exceptions in an efficient way if required to.

- Execution wind-down. This should release any resource required by
  the gufunc to execute, as well as collect and forward the exception
  information it may have occured.


Note that while resolving, a "kernel" must be selected. We do not want
to limit us to preexisting loops, so there should be a hook allowing
able to generate a specific kernel. Allowing to have gufuncs that have
no kernel associated, but that generates them as needed. The interface
should be such as to support tools like numba.

Caveats
=======

Exception handling
------------------

In NumPy's gufuncs, exception handling is minimal. The gufunc
machinery can detect that something failed, and typically will either
ignore, warn or raise an exception if something went wrong.

There are little support for error recovery. Typically, a error will
be signaled by placing an invalid value as the result (NaN for
floating point). Then NumPy will react according to the set policy on
NaN, so it may ignore, warn or raise on its detection. This means that
taking recovery actions is tricky:

- On ignore or warn: The output will need to be checked for those
  invalid values in order to take corrective action. There is no way
  to programmatically distinguish between a complete, correct run from
  a run that had errors on some of the elements.

- On raise: the partial output is typically discarded and there is no
  information about which elements produced an invalid result.

A vectorized execution may have some of the elements producing a
condition that should raise but, at the same time, produce the proper
result for a large set of elements.

In an ideal world, the gufunc should have three different results from
the point of view of exception handling:

a. Total success. All elements were able to execute without producing
   an exceptional situation. No exception should be raised.

b. Partial success. Some elements were able to execute without
   producing an exceptional situation. An exception should be raised,
   but the exception should provide enough information so that:
   - It is possible to retrieve the results from the elements that
     didn't raise a exception.
   - There is information about which elements finished properly and
     which elements raised.

c. Failure. Either the gufunc wasn't able to set up (i.e. arguments not
   matching the gufunc description) or all elements ended raising an
   exception. Not that the latter could also be handled as a partial
   success with all elements marked as "raising".

Exceptions from partial success and failures should be distinguishable.
With the extra information many different recovery strategies could be
implemented.


Parallelization and Semantics
-----------------------------

There are some important decissions to be made on semantics of
execution. The most important being what semantics should the system
provide regarding to kernel execution ordering when dealing with
mutable inputs (including input-output aliasing).

In general, in order to be able to parallelize the execution, no
ordering should be guaranteed. Execution for any element should be
indepent from other elements. Inputs should act always as "read-only"
and a kernel shouldn't be able to read any output.

At most, we could support "input-output" arguments but enforcing that
the kernel can only access its element.

Other possible ideas would be wrt parallelization would be:
- future-like output arguments, that will execute the gufunc async to the
  calling code.

- lazy-like calling, where the gufunc performs some basic preparations
  but execution is delayed to build some sort of expression tree.

Note that all this has implications on argument semantics, as
arguments should be "captured" when the gufunc is called (and not on
its execution). Enforcing this could be difficult, while having that
documented may result in behavior that is not expected by many.


Reductions
----------

TBD.
