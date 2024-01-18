from evaluate import pred
from preprocessing import TensorBSplineTransformer, DeltaEOptimizer, NLOptOptimizer, PolynomialTransformer, GAMOptimizer
import numpy as np
import colour
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from utils import compute_response

import pandas as pd

SFU_FILE_PATH = 'reflect_db.reflect'
CMF_RANGE = colour.SpectralShape(400, 700, 10)
ILLUMINANT = "D65"
NIKON = 'Nikon 5100 (NPL)'
SIGMA = 'Sigma SDMerill (NPL)'
TEST_SIZE = 0.1
RANDOM_STATE = 0

cmfs = colour.colorimetry.MSDS_CMFS_STANDARD_OBSERVER['CIE 1931 2 Degree Standard Observer'].align(CMF_RANGE)
nikon = colour.MSDS_CAMERA_SENSITIVITIES[NIKON].align(CMF_RANGE)
sigma = colour.MSDS_CAMERA_SENSITIVITIES[SIGMA].align(CMF_RANGE)
macbeth = colour.MultiSpectralDistributions(colour.SDS_COLOURCHECKERS['babel_average']).align(CMF_RANGE)
D65 = colour.SDS_ILLUMINANTS[ILLUMINANT].align(CMF_RANGE)
response_human_macbeth = colour.characterisation.training_data_sds_to_XYZ(macbeth, cmfs, D65, None)
nikon_d65 = colour.characterisation.normalise_illuminant(D65, nikon)
sigma_d65 = colour.characterisation.normalise_illuminant(D65, sigma)
response_nikon = compute_response(macbeth, D65, nikon, wb=True)
response_sigma = compute_response(macbeth, D65, sigma, wb=True)

macbeth_out_srgb = colour.XYZ_to_sRGB(response_human_macbeth)
nikon_out_srgb = colour.cctf_encoding(response_nikon)
sigma_out_srgb = colour.cctf_encoding(response_sigma)

fig, ax = colour.plotting.plot_multi_cmfs([colour.MSDS_CAMERA_SENSITIVITIES[SIGMA], cmfs], show=False)
plt.title("Sigma SD1 Merrill (lens unknown) Spectral Sensitivities", weight="bold", fontsize="15")
ax.set_xlabel('Wavelength', weight="bold", fontsize="12")
ax.set_ylabel("Sensitivity", weight="bold", fontsize="12")
ax.legend()
plt.show(block=True)
fig, ax  =colour.plotting.plot_multi_cmfs(colour.MSDS_CAMERA_SENSITIVITIES[NIKON], show=False)
plt.title("Nikon D5100 (lens unknown) Spectral Sensitivities", weight="bold", fontsize="15")
ax.set_xlabel('Wavelength', weight="bold", fontsize="12")
ax.set_ylabel("Sensitivity", weight="bold", fontsize="12")
ax.legend()
plt.show(block=True)


colour.plotting.plot_multi_colour_swatches(macbeth_out_srgb, columns=6, direction="-y", spacing=0.05, legend=False, show=False)
plt.show(block=True)
colour.plotting.plot_multi_colour_swatches(nikon_out_srgb, columns=6, direction="-y", spacing=0.05, legend=False, show=False)
plt.show(block=True)
colour.plotting.plot_multi_colour_swatches(sigma_out_srgb, columns=6, direction="-y", spacing=0.05, legend=False, show=False)
plt.show(block=True)
