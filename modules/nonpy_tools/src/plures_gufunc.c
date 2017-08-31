/*
 * This code is inspired in the ufunc_object code for generalized ufuncs.
 * PyUFunc_GeneralizedFunction.
 */

typedef plures_array_def {
    plures_datashape datashape;
    void *data;
    /* here should go an abstract interface to make sense of the data */
} plures_array;


typedef plures_gufunc_def {
    plures_parsed_signature *signature;
    plures_coreloop_map *coreloop_map;
} plures_gufunc;

/* originally, the function handling mechanism takes many 
 * Python/NumPy specific arguments. The first action would
 * be to refactor those so that a clean interface for the
 * call function that is Python independant exists.
 */

static int
PyUFunc_GeneralizedFunction(plures_gufunc *ufunc,
                            plures_array *args,
                            plures_gufunc_options *options,
                            PyArrayObject **op)
{
    int nin, nout;
    int i, j, idim, nop;
    const char *ufunc_name;
    int retval = -1, subok = 1;
    int needs_api = 0;

    PyArray_Descr *dtypes[NPY_MAXARGS];

    /* Use remapped axes for generalized ufunc */
    int broadcast_ndim, iter_ndim;
    int op_axes_arrays[NPY_MAXARGS][NPY_MAXDIMS];
    int *op_axes[NPY_MAXARGS];

    npy_uint32 op_flags[NPY_MAXARGS];
    npy_intp iter_shape[NPY_MAXARGS];
    NpyIter *iter = NULL;
    npy_uint32 iter_flags;
    npy_intp total_problem_size;

    /* These parameters come from extobj= or from a TLS global */
    int buffersize = 0, errormask = 0;

    /* The selected inner loop */
    PyUFuncGenericFunction innerloop = NULL;
    void *innerloopdata = NULL;
    /* The dimensions which get passed to the inner loop */
    npy_intp inner_dimensions[NPY_MAXDIMS+1];
    /* The strides which get passed to the inner loop */
    npy_intp *inner_strides = NULL;

    /* The sizes of the core dimensions (# entries is ufunc->core_num_dim_ix) */
    npy_intp *core_dim_sizes = inner_dimensions + 1;
    int core_dim_ixs_size;

    /* The __array_prepare__ function to call for each output */
    PyObject *arr_prep[NPY_MAXARGS];
    /*
     * This is either args, or args with the out= parameter from
     * kwds added appropriately.
     */
    PyObject *arr_prep_args = NULL;

    NPY_ORDER order = NPY_KEEPORDER;
    /* Use the default assignment casting rule */
    NPY_CASTING casting = NPY_DEFAULT_ASSIGN_CASTING;
    /* When provided, extobj and typetup contain borrowed references */
    PyObject *extobj = NULL, *type_tup = NULL;

    if (ufunc == NULL) {
        PyErr_SetString(PyExc_ValueError, "function not supported");
        return -1;
    }

    nin = ufunc->nin;
    nout = ufunc->nout;
    nop = nin + nout;

    ufunc_name = ufunc->name ? ufunc->name : "<unnamed ufunc>";

    NPY_UF_DBG_PRINT1("\nEvaluating ufunc %s\n", ufunc_name);

    /* Initialize all the operands and dtypes to NULL */
    for (i = 0; i < nop; ++i) {
        op[i] = NULL;
        dtypes[i] = NULL;
        arr_prep[i] = NULL;
    }

    NPY_UF_DBG_PRINT("Getting arguments\n");

    /* Get all the arguments */
    retval = get_ufunc_arguments(ufunc, args, kwds,
                op, &order, &casting, &extobj,
                &type_tup, &subok, NULL);
    if (retval < 0) {
        goto fail;
    }

    /*
     * Figure out the number of iteration dimensions, which
     * is the broadcast result of all the input non-core
     * dimensions.
     */
    broadcast_ndim = 0;
    for (i = 0; i < nin; ++i) {
        int n = PyArray_NDIM(op[i]) - ufunc->core_num_dims[i];
        if (n > broadcast_ndim) {
            broadcast_ndim = n;
        }
    }

    /*
     * Figure out the number of iterator creation dimensions,
     * which is the broadcast dimensions + all the core dimensions of
     * the outputs, so that the iterator can allocate those output
     * dimensions following the rules of order='F', for example.
     */
    iter_ndim = broadcast_ndim;
    for (i = nin; i < nop; ++i) {
        iter_ndim += ufunc->core_num_dims[i];
    }
    if (iter_ndim > NPY_MAXDIMS) {
        PyErr_Format(PyExc_ValueError,
                    "too many dimensions for generalized ufunc %s",
                    ufunc_name);
        retval = -1;
        goto fail;
    }

    /*
     * Validate the core dimensions of all the operands, and collect all of
     * the labelled core dimensions into 'core_dim_sizes'.
     *
     * The behavior has been changed in NumPy 1.10.0, and the following
     * requirements must be fulfilled or an error will be raised:
     *  * Arguments, both input and output, must have at least as many
     *    dimensions as the corresponding number of core dimensions. In
     *    previous versions, 1's were prepended to the shape as needed.
     *  * Core dimensions with same labels must have exactly matching sizes.
     *    In previous versions, core dimensions of size 1 would broadcast
     *    against other core dimensions with the same label.
     *  * All core dimensions must have their size specified by a passed in
     *    input or output argument. In previous versions, core dimensions in
     *    an output argument that were not specified in an input argument,
     *    and whose size could not be inferred from a passed in output
     *    argument, would have their size set to 1.
     */
    for (i = 0; i < ufunc->core_num_dim_ix; ++i) {
        core_dim_sizes[i] = -1;
    }
    for (i = 0; i < nop; ++i) {
        if (op[i] != NULL) {
            int dim_offset = ufunc->core_offsets[i];
            int num_dims = ufunc->core_num_dims[i];
            int core_start_dim = PyArray_NDIM(op[i]) - num_dims;

            /* Check if operands have enough dimensions */
            if (core_start_dim < 0) {
                PyErr_Format(PyExc_ValueError,
                        "%s: %s operand %d does not have enough "
                        "dimensions (has %d, gufunc core with "
                        "signature %s requires %d)",
                        ufunc_name, i < nin ? "Input" : "Output",
                        i < nin ? i : i - nin, PyArray_NDIM(op[i]),
                        ufunc->core_signature, num_dims);
                retval = -1;
                goto fail;
            }

            /*
             * Make sure every core dimension exactly matches all other core
             * dimensions with the same label.
             */
            for (idim = 0; idim < num_dims; ++idim) {
                int core_dim_index = ufunc->core_dim_ixs[dim_offset+idim];
                npy_intp op_dim_size =
                            PyArray_DIM(op[i], core_start_dim+idim);

                if (core_dim_sizes[core_dim_index] == -1) {
                    core_dim_sizes[core_dim_index] = op_dim_size;
                }
                else if (op_dim_size != core_dim_sizes[core_dim_index]) {
                    PyErr_Format(PyExc_ValueError,
                            "%s: %s operand %d has a mismatch in its "
                            "core dimension %d, with gufunc "
                            "signature %s (size %zd is different "
                            "from %zd)",
                            ufunc_name, i < nin ? "Input" : "Output",
                            i < nin ? i : i - nin, idim,
                            ufunc->core_signature, op_dim_size,
                            core_dim_sizes[core_dim_index]);
                    retval = -1;
                    goto fail;
                }
            }
        }
    }

    /*
     * Make sure no core dimension is unspecified.
     */
    for (i = 0; i < ufunc->core_num_dim_ix; ++i) {
        if (core_dim_sizes[i] == -1) {
            break;
        }
    }
    if (i != ufunc->core_num_dim_ix) {
        /*
         * There is at least one core dimension missing, find in which
         * operand it comes up first (it has to be an output operand).
         */
        const int missing_core_dim = i;
        int out_op;
        for (out_op = nin; out_op < nop; ++out_op) {
            int first_idx = ufunc->core_offsets[out_op];
            int last_idx = first_idx + ufunc->core_num_dims[out_op];
            for (i = first_idx; i < last_idx; ++i) {
                if (ufunc->core_dim_ixs[i] == missing_core_dim) {
                    break;
                }
            }
            if (i < last_idx) {
                /* Change index offsets for error message */
                out_op -= nin;
                i -= first_idx;
                break;
            }
        }
        PyErr_Format(PyExc_ValueError,
                     "%s: Output operand %d has core dimension %d "
                     "unspecified, with gufunc signature %s",
                     ufunc_name, out_op, i, ufunc->core_signature);
        retval = -1;
        goto fail;
    }

    /* Fill in the initial part of 'iter_shape' */
    for (idim = 0; idim < broadcast_ndim; ++idim) {
        iter_shape[idim] = -1;
    }

    /* Fill in op_axes for all the operands */
    j = broadcast_ndim;
    core_dim_ixs_size = 0;
    for (i = 0; i < nop; ++i) {
        int n;
        if (op[i]) {
            /*
             * Note that n may be negative if broadcasting
             * extends into the core dimensions.
             */
            n = PyArray_NDIM(op[i]) - ufunc->core_num_dims[i];
        }
        else {
            n = broadcast_ndim;
        }
        /* Broadcast all the unspecified dimensions normally */
        for (idim = 0; idim < broadcast_ndim; ++idim) {
            if (idim >= broadcast_ndim - n) {
                op_axes_arrays[i][idim] = idim - (broadcast_ndim - n);
            }
            else {
                op_axes_arrays[i][idim] = -1;
            }
        }

        /* Any output core dimensions shape should be ignored */
        for (idim = broadcast_ndim; idim < iter_ndim; ++idim) {
            op_axes_arrays[i][idim] = -1;
        }

        /* Except for when it belongs to this output */
        if (i >= nin) {
            int dim_offset = ufunc->core_offsets[i];
            int num_dims = ufunc->core_num_dims[i];
            /* Fill in 'iter_shape' and 'op_axes' for this output */
            for (idim = 0; idim < num_dims; ++idim) {
                iter_shape[j] = core_dim_sizes[
                                        ufunc->core_dim_ixs[dim_offset + idim]];
                op_axes_arrays[i][j] = n + idim;
                ++j;
            }
        }

        op_axes[i] = op_axes_arrays[i];
        core_dim_ixs_size += ufunc->core_num_dims[i];
    }

    /* Get the buffersize and errormask */
    if (_get_bufsize_errmask(extobj, ufunc_name, &buffersize, &errormask) < 0) {
        retval = -1;
        goto fail;
    }

    NPY_UF_DBG_PRINT("Finding inner loop\n");


    retval = ufunc->type_resolver(ufunc, casting,
                            op, type_tup, dtypes);
    if (retval < 0) {
        goto fail;
    }
    /* For the generalized ufunc, we get the loop right away too */
    retval = ufunc->legacy_inner_loop_selector(ufunc, dtypes,
                                    &innerloop, &innerloopdata, &needs_api);
    if (retval < 0) {
        goto fail;
    }

#if NPY_UF_DBG_TRACING
    printf("input types:\n");
    for (i = 0; i < nin; ++i) {
        PyObject_Print((PyObject *)dtypes[i], stdout, 0);
        printf(" ");
    }
    printf("\noutput types:\n");
    for (i = nin; i < nop; ++i) {
        PyObject_Print((PyObject *)dtypes[i], stdout, 0);
        printf(" ");
    }
    printf("\n");
#endif

    if (subok) {
        /*
         * Get the appropriate __array_prepare__ function to call
         * for each output
         */
        _find_array_prepare(args, kwds, arr_prep, nin, nout, 0);

        /* Set up arr_prep_args if a prep function was needed */
        for (i = 0; i < nout; ++i) {
            if (arr_prep[i] != NULL && arr_prep[i] != Py_None) {
                arr_prep_args = make_arr_prep_args(nin, args, kwds);
                break;
            }
        }
    }

    /* If the loop wants the arrays, provide them */
    if (_does_loop_use_arrays(innerloopdata)) {
        innerloopdata = (void*)op;
    }

    /*
     * Set up the iterator per-op flags.  For generalized ufuncs, we
     * can't do buffering, so must COPY or UPDATEIFCOPY.
     */
    for (i = 0; i < nin; ++i) {
        op_flags[i] = NPY_ITER_READONLY |
                      NPY_ITER_COPY |
                      NPY_ITER_ALIGNED;
        /*
         * If READWRITE flag has been set for this operand,
         * then clear default READONLY flag
         */
        op_flags[i] |= ufunc->op_flags[i];
        if (op_flags[i] & (NPY_ITER_READWRITE | NPY_ITER_WRITEONLY)) {
            op_flags[i] &= ~NPY_ITER_READONLY;
        }
    }
    for (i = nin; i < nop; ++i) {
        op_flags[i] = NPY_ITER_READWRITE|
                      NPY_ITER_UPDATEIFCOPY|
                      NPY_ITER_ALIGNED|
                      NPY_ITER_ALLOCATE|
                      NPY_ITER_NO_BROADCAST;
    }

    iter_flags = ufunc->iter_flags |
                 NPY_ITER_MULTI_INDEX |
                 NPY_ITER_REFS_OK |
                 NPY_ITER_REDUCE_OK |
                 NPY_ITER_ZEROSIZE_OK;

    /* Create the iterator */
    iter = NpyIter_AdvancedNew(nop, op, iter_flags,
                           order, NPY_UNSAFE_CASTING, op_flags,
                           dtypes, iter_ndim,
                           op_axes, iter_shape, 0);
    if (iter == NULL) {
        retval = -1;
        goto fail;
    }

    /* Fill in any allocated outputs */
    for (i = nin; i < nop; ++i) {
        if (op[i] == NULL) {
            op[i] = NpyIter_GetOperandArray(iter)[i];
            Py_INCREF(op[i]);
        }
    }

    /*
     * Set up the inner strides array. Because we're not doing
     * buffering, the strides are fixed throughout the looping.
     */
    inner_strides = (npy_intp *)PyArray_malloc(
                        NPY_SIZEOF_INTP * (nop+core_dim_ixs_size));
    if (inner_strides == NULL) {
        PyErr_NoMemory();
        retval = -1;
        goto fail;
    }
    /* Copy the strides after the first nop */
    idim = nop;
    for (i = 0; i < nop; ++i) {
        int num_dims = ufunc->core_num_dims[i];
        int core_start_dim = PyArray_NDIM(op[i]) - num_dims;
        /*
         * Need to use the arrays in the iterator, not op, because
         * a copy with a different-sized type may have been made.
         */
        PyArrayObject *arr = NpyIter_GetOperandArray(iter)[i];
        npy_intp *shape = PyArray_SHAPE(arr);
        npy_intp *strides = PyArray_STRIDES(arr);
        for (j = 0; j < num_dims; ++j) {
            if (core_start_dim + j >= 0) {
                /*
                 * Force the stride to zero when the shape is 1, sot
                 * that the broadcasting works right.
                 */
                if (shape[core_start_dim + j] != 1) {
                    inner_strides[idim++] = strides[core_start_dim + j];
                } else {
                    inner_strides[idim++] = 0;
                }
            } else {
                inner_strides[idim++] = 0;
            }
        }
    }

    total_problem_size = NpyIter_GetIterSize(iter);
    if (total_problem_size < 0) {
        /*
         * Only used for threading, if negative (this means that it is
         * larger then ssize_t before axes removal) assume that the actual
         * problem is large enough to be threaded usefully.
         */
        total_problem_size = 1000;
    }

    /* Remove all the core output dimensions from the iterator */
    for (i = broadcast_ndim; i < iter_ndim; ++i) {
        if (NpyIter_RemoveAxis(iter, broadcast_ndim) != NPY_SUCCEED) {
            retval = -1;
            goto fail;
        }
    }
    if (NpyIter_RemoveMultiIndex(iter) != NPY_SUCCEED) {
        retval = -1;
        goto fail;
    }
    if (NpyIter_EnableExternalLoop(iter) != NPY_SUCCEED) {
        retval = -1;
        goto fail;
    }

    /*
     * The first nop strides are for the inner loop (but only can
     * copy them after removing the core axes
     */
    memcpy(inner_strides, NpyIter_GetInnerStrideArray(iter),
                                    NPY_SIZEOF_INTP * nop);

#if 0
    printf("strides: ");
    for (i = 0; i < nop+core_dim_ixs_size; ++i) {
        printf("%d ", (int)inner_strides[i]);
    }
    printf("\n");
#endif

    /* Start with the floating-point exception flags cleared */
    PyUFunc_clearfperr();

    NPY_UF_DBG_PRINT("Executing inner loop\n");

    if (NpyIter_GetIterSize(iter) != 0) {
        /* Do the ufunc loop */
        NpyIter_IterNextFunc *iternext;
        char **dataptr;
        npy_intp *count_ptr;
        NPY_BEGIN_THREADS_DEF;

        /* Get the variables needed for the loop */
        iternext = NpyIter_GetIterNext(iter, NULL);
        if (iternext == NULL) {
            retval = -1;
            goto fail;
        }
        dataptr = NpyIter_GetDataPtrArray(iter);
        count_ptr = NpyIter_GetInnerLoopSizePtr(iter);

        if (!needs_api && !NpyIter_IterationNeedsAPI(iter)) {
            NPY_BEGIN_THREADS_THRESHOLDED(total_problem_size);
        }
        do {
            inner_dimensions[0] = *count_ptr;
            innerloop(dataptr, inner_dimensions, inner_strides, innerloopdata);
        } while (iternext(iter));

        if (!needs_api && !NpyIter_IterationNeedsAPI(iter)) {
            NPY_END_THREADS;
        }
    } else {
        /**
         * For each output operand, check if it has non-zero size,
         * and assign the identity if it does. For example, a dot
         * product of two zero-length arrays will be a scalar,
         * which has size one.
         */
        for (i = nin; i < nop; ++i) {
            if (PyArray_SIZE(op[i]) != 0) {
                switch (ufunc->identity) {
                    case PyUFunc_Zero:
                        assign_reduce_identity_zero(op[i], NULL);
                        break;
                    case PyUFunc_One:
                        assign_reduce_identity_one(op[i], NULL);
                        break;
                    case PyUFunc_MinusOne:
                        assign_reduce_identity_minusone(op[i], NULL);
                        break;
                    case PyUFunc_None:
                    case PyUFunc_ReorderableNone:
                        PyErr_Format(PyExc_ValueError,
                                "ufunc %s ",
                                ufunc_name);
                        retval = -1;
                        goto fail;
                    default:
                        PyErr_Format(PyExc_ValueError,
                                "ufunc %s has an invalid identity for reduction",
                                ufunc_name);
                        retval = -1;
                        goto fail;
                }
            }
        }
    }

    /* Check whether any errors occurred during the loop */
    if (PyErr_Occurred() ||
        _check_ufunc_fperr(errormask, extobj, ufunc_name) < 0) {
        retval = -1;
        goto fail;
    }

    PyArray_free(inner_strides);
    NpyIter_Deallocate(iter);
    /* The caller takes ownership of all the references in op */
    for (i = 0; i < nop; ++i) {
        Py_XDECREF(dtypes[i]);
        Py_XDECREF(arr_prep[i]);
    }
    Py_XDECREF(type_tup);
    Py_XDECREF(arr_prep_args);

    NPY_UF_DBG_PRINT("Returning Success\n");

    return 0;

fail:
    NPY_UF_DBG_PRINT1("Returning failure code %d\n", retval);
    PyArray_free(inner_strides);
    NpyIter_Deallocate(iter);
    for (i = 0; i < nop; ++i) {
        Py_XDECREF(op[i]);
        op[i] = NULL;
        Py_XDECREF(dtypes[i]);
        Py_XDECREF(arr_prep[i]);
    }
    Py_XDECREF(type_tup);
    Py_XDECREF(arr_prep_args);

    return retval;
}

