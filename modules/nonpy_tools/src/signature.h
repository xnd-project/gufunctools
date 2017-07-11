#ifndef GUFT_SIGNATURE_H
#define GUFT_SIGNATURE_H

typedef struct _parsed_signature_header_struct {
    size_t input_count;
    size_t output_count;
    size_t arg_count;
    size_t dimension_variable_count;
    size_t total_signature_dimensions;
    size_t *arg_dimension_count; /* as many arg_count */
    size_t *arg_shape_offsets; /* as many arg_count */
    size_t *arg_shape_idx; /* as many as total_signature_dimensions */

    /* the next is the start to the variable length data pointed by the above
       members */
    size_t data[]; 
} parsed_signature;


/* legacy function that requires nin and nargs just as NumPy internal code */
parsed_signature *
legacy_numpy_parse_signature(const char *signature, int nin, int nargs);

/* like legacy_numpy_parse_signature but it figures out nin and nargs from
   the signature */
parsed_signature *
numpy_parse_signature(const char *signature);

void
print_parsed_signature(parsed_signature *the_signature);

void
dispose_parsed_signature(parsed_signature *the_signature);


#endif /* GUFT_SIGNATURE_H */
