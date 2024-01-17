from colour import apply_matrix_colour_correction

def interleave_arrays(array1, array2):
    # This function assumes both arrays have the same length
    interleaved = []
    for a, b in zip(array1, array2):
        interleaved.extend([a, b])
    return interleaved

def create_matrix_correction_func(method="Finlayson 2015", degree=3, root_polynomial_expansion=True):
    def matrix_correction_func(a, M):
        return apply_matrix_colour_correction(
            a,
            M,
            method=method,
            degree=degree,
            root_polynomial_expansion=root_polynomial_expansion,
        )
    return matrix_correction_func
