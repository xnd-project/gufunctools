"""A simple gufunc like object"""


"""
def sample_resolver(parsed_signature, input_shapes, output_shapes):
    nin, nout, nargs = parsed_signature[0]
    sig_shapes = parsed_signature[2]
    sig_shapes_len = [len(sh) for sh in sig_shapes]
    input_shape_count = len(input_shapes)
    output_shape_count = len(output_shapes)

    if input_shape_count != nin:
        raise RuntimeError("Number of input arguments mismatch")

    if output_shape_count > nout:
        raise RuntimeError("Too many explicit output arguments")

    dim_vars =  [None] * parsed_signature[1]

    concrete = []
    iter_shapes = []
    # accumulate shapes for inputs
    for _input in range(nin):
        abstract_shape = sig_shapes[_input]
        dim_count = len(abstract_shape)
        concrete_shape = input_shapes[-dim_count:]
        for _dim in range(dim_count):
            val = _concrete_shape[_dim]
            dim_var_idx = _abstract_shape[_dim]
            if dim_vars[dim_var_idx] == None:
                dim_vars[dim_var_idx] = val
            elif dim_vars[dim_var_idx] != val:
                raise RuntimeError("Argument shapes incompatible with signature")

    # At this point... we should be able to call a hook for extra validation
    # resolving
    resolve_hook = getattr(self, '_resolve_hook', None)
    if resolve_hook:
        resolve_hook(dim_vars)

    # validate the explicit ouputs
    for _output in range(nin, nin+output_shape_count):
        abstract_shape = sig_shape[_output]
        dim_count = len(abstract_shape)
        concrete_shape = 
        for _dim in range(dim_count):
            val = _concrete_shape[_dim]
            dim_var_idx = _abstract_shape[_dim]
            if dim_vars[dim_var_idx] == None:
                dim_vars[dim_var_idx] = val
            elif dim_vars[dim_var_idx] != val:
                raise RuntimeError("Argument shapes incompatible with signature")

            

    # accumulate shapes for explicit outputs
    for i in range(output_shape_count):
        concrete.expand(output_shapes[i][-sig_shapes[i+nin]:])

    # fill remaining outputs

    shapes_flattened = [e for t in shapes for e in t]


    concrete = []

    # add inputs
    for i in range(len(input_shapes)):
        

    all_shapes = input_shapes+output_shapes
    concrete = input_shapes + output_shapes
    concrete_flattened = [e for t in actual for e in t]

    # build (and check) the variables from the concrete arrays:
    for i in len(concrete_flattened):
        
    
    actual_flattened.extend([None]


    #this is made too difficult by not having
"""


'''
class gufunc(Object):
    def __init__(self, signature, resolve_fn, kernel_fn):
        self.parsed_signature = parse_signature(signature)
        self._resolver = resolve_fn
        self._kernel = kernel_fn


    def __call__(*args, **kwargs):

        # from args it should be possible to obtain the outer_dimension and
        # the full set of dimension variables match.
        inputs= [adapt_input(arg) for arg in args]
        outputs = kwargs.pop('output')
        outputs = [adapt_output(arg) for arg in outputs]

        input_dims = [array.shape for array in args]
        arg_types = [array.types for array in args]
        try:
            # resolved data should provide:
            #
            # - iteration shape
            #
            # - resolved inner dimensions of the outputs
            resolved_data = resolve(self.parsed_signature,
                                    input_shapes,
                                    output_shapes)
        except Exception:
            # could not resolve shapes. This may be due to:
            # 
            # - Inner dimensions can not be matched
            #   (incompatible with the signature)
            #
            # - Incompatible shapes in the outer dimensions of the arguments
            #   (can not make an iteration compatible with the shapes, this
            #   should, according to broadcasting rules)
            raise

        # pick kernel that is appropriate for the types.
        #
        # kernel must contain the code that must actually be called with a known
        # interface.
        try:
            kernel = self.pick_kernel(arg_types)
        except Exception:
            # This could fail if there isn't a valid kernel for shape
            raise

        # At this point we know that we can actually iterate to generate
        # results, now check the outputs:
        try:
            # from resolved_data the following items would be required:
            #
            # - inner shapes of outputs
            #
            # - types of the outputs
            #
            # - iteration shape
            #
            # The result will hold the arrays to be used as outputs. If those
            # where explicitly specified, they have been checked to have the
            # right size (iteration_shape ::: inner_shape from signature).
            outputs = output_check_or_create(outputs, resolved_data,
                                             kernel.output_types)
        except Exception:
            # this could fail due to:
            #
            # - Explicit outputs were provided that did not match the required
            #   shapes
            #
            # - MemoryError due to out of memory is also possible
            raise

        # this should build some kind of iterator and launch the code in the
        # kernel over the argument-outputs.

        try:
            execute_kernel(resolved_data.iteration_shape,
                           kernel,
                           args + outputs);
        except Exception:
            # Possible fails include:
            #
            # - Failure on all elements (total failure)
            #
            # - Failure on some of the elements (partial failure)
            #
            # There should be a policy on how to deal with partial failure.  In
            # general, in partial failure result data should be available and
            # extra information about which elements failed should be
            # present, so that corrective action could be taken. This could take
            # the form of a mask or of an array of indices indicating failures.
            #
            # It could also be interesting providing exception information of
            # the failures. That could be just an array with the generated
            # exceptions. Array of indices could then be 
            raise
        pass

'''
