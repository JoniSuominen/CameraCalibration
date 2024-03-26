from colour import apply_matrix_colour_correction
from scipy.stats import poisson
from numpy.random import default_rng
import numpy as np

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

def add_noise(raw, poisson_gain, gaussian_gain):

    # Initialize random number generator
    rng = default_rng()

    if poisson_gain == 0:  # no Poissonian component
        z = raw
    else:  # Poissonian component
        chi = 1 / poisson_gain**2
        z = poisson.rvs(np.maximum(0, chi * raw)) / chi

    # Gaussian component
    z = z + np.sqrt(gaussian_gain**2) * rng.standard_normal(raw.shape)

    # CLIPPING
    z = np.maximum(0,np.minimum(z, 1))

    return z