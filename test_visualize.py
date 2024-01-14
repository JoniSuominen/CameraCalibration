import colour
import numpy as np
import matplotlib.pyplot as plt
from utils import compute_response, deltae_mean, parse_reflectance_spectra, interleave_arrays
import pandas as pd
from sklearn.model_selection import train_test_split

CMF_RANGE = colour.SpectralShape(400, 700, 10)
SFU_FILE_PATH = 'reflect_db.reflect'
TEST_SIZE = 0.2
RANDOM_STATE = 0

def load_spectral_data(csv_file_path):
    """
    Load spectral data from a CSV file.
    The first column is assumed to be non-spectral (e.g., an identifier).
    """
    df = pd.read_csv(csv_file_path)
    spectral_data = df.iloc[:, :].to_numpy()
    return spectral_data

reflectance_spectra, _ = parse_reflectance_spectra(SFU_FILE_PATH)
csv_file_path = 'downsampled_spectral_data.csv'  # Replace with your CSV file path
spectral_data = load_spectral_data(csv_file_path)


test_foster = colour.MultiSpectralDistributions(spectral_data.T, CMF_RANGE).align(CMF_RANGE)
test_sfu = colour.MultiSpectralDistributions(reflectance_spectra).align(CMF_RANGE)

# CIE 1931 2 Degree Standard Observer Color Matching Functions
cmfs = colour.colorimetry.MSDS_CMFS_STANDARD_OBSERVER['CIE 1931 2 Degree Standard Observer'].align(CMF_RANGE)

# D65 illuminant SPD
illuminant = colour.SDS_ILLUMINANTS['D65'].align(CMF_RANGE)

# Convert hyperspectral image data to XYZ values
XYZ_foster = colour.msds_to_XYZ(test_foster, cmfs, illuminant, method="Integration", shape=CMF_RANGE)
XYZ_sfu = colour.msds_to_XYZ(test_sfu, cmfs, illuminant, method="Integration", shape=CMF_RANGE)

XYZ_combined = np.vstack((XYZ_foster, XYZ_sfu))
mock_responses = np.ones(XYZ_foster.shape)

X_train, X_test, y_train, y_test = train_test_split(XYZ_foster, mock_responses, test_size=TEST_SIZE, random_state=RANDOM_STATE)

# Convert XYZ to xy chromaticity coordinates
xy_foster = colour.XYZ_to_xy(X_test)
xy_sfu = colour.XYZ_to_xy(XYZ_sfu)
xy_combined = colour.XYZ_to_xy(X_test)

# Plotting on the CIE 1931 chromaticity diagram
plot, axes = colour.plotting.plot_chromaticity_diagram_CIE1931(standalone=False)
axes.plot(xy_combined[..., 0], xy_combined[..., 1], 'o', color='red', markeredgecolor='black')
# axes.plot(xy_sfu[..., 0], xy_sfu[..., 1], 'o', color='blue', markeredgecolor='black')

plt.title("SFU Chromaticity Coordinates on the CIE 1931 Chromaticity Diagram")
plt.show(block=True)
