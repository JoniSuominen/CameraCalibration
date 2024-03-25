import pandas as pd
import numpy as np
import torch

from colour import MultiSpectralDistributions, read_sds_from_csv_file, MSDS_CAMERA_SENSITIVITIES, SpectralShape, read_spectral_data_from_csv_file
from colour.colorimetry import sds_and_msds_to_msds, MSDS_CMFS_STANDARD_OBSERVER, SDS_ILLUMINANTS
from colour.characterisation import normalise_illuminant

NIKON = 'Nikon 5100 (NPL)'
SIGMA = 'Sigma SDMerill (NPL)'
CANON_PATH = "CANON.csv"
ILLUMINANT = "D65"
CMF_RANGE = SpectralShape(400, 700, 10)
INTERP_CMF_RANGE = SpectralShape(400,700,10)

def load_insitu(csv_file_path):
    """
    Load spectral data from a CSV file.
    The first column is assumed to be non-spectral (e.g., an identifier).
    The rest of the columns are assumed to be normalized reflectances with sampling 400:10:700 nm.
    """
    df = pd.read_csv(csv_file_path)
    wavelengths  = df.iloc[:, 0].to_numpy()
    values  = df.iloc[:, 1:].to_numpy()
    return MultiSpectralDistributions(values, wavelengths).align(INTERP_CMF_RANGE)


def msds_to_xyz(reflectances, sensitivities, illuminant):
    response = (sensitivities.values.T @ (np.diag(illuminant.values) @ reflectances.values)).T
    max_response = compute_response_perfect_reflector(sensitivities)
    print(max_response / max_response[1])
    return response / max_response[1]

def msds_to_rgb(reflectances, sensitivities, illuminant):
    response = (sensitivities.values.T @ (np.diag(illuminant.values) @ reflectances.values)).T
    max_response = compute_response_perfect_reflector(sensitivities)
    return response / max_response  


def compute_response_perfect_reflector(sensitivities):
    illuminant = SDS_ILLUMINANTS["D65"].align(INTERP_CMF_RANGE)
    response_test = (sensitivities.values.T @ np.diag(illuminant.values)).T
    response_sum = np.sum(response_test, axis=0)
    
    return response_sum


def load_dataset_csv(csv_file_path):
    """
    Load spectral data from a CSV file.
    The first column is assumed to be non-spectral (e.g., an identifier).
    The rest of the columns are assumed to be normalized reflectances with sampling 400:10:700 nm.
    """
    df = pd.read_csv(csv_file_path)
    spectral_data = df.iloc[:, 1:].to_numpy()
    return MultiSpectralDistributions(spectral_data.T, CMF_RANGE).align(INTERP_CMF_RANGE)


def load_dataset_skin():
    """
    Load spectral data from a CSV file.
    The first column is assumed to be non-spectral (e.g., an identifier).
    The rest of the columns are assumed to be normalized reflectances with sampling 400:10:700 nm.
    """
    skindata = read_sds_from_csv_file("Skin.csv")
    return MultiSpectralDistributions(skindata).align(INTERP_CMF_RANGE)


def load_dataset_sfu(file_path):
    reflectance_spectra = []
    current_spectra = []

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()  # Remove leading and trailing whitespaces
            if line.startswith('#'):  # Ignore comments
                continue
            elif line == "":  # New line indicates a new object
                if current_spectra:  # If there is data, add it to the list
                    reflectance_spectra.append(current_spectra)
                    current_spectra = []
            else:
                current_spectra.append(float(line))  # Add the reflectance value to the current spectra

        # Add the last spectra if there is any
        if current_spectra:
            reflectance_spectra.append(current_spectra)
            
    reflectance_spectra = np.array(reflectance_spectra)
            
    wavelengths =  np.linspace(380, 780, 101)
    
    data = {wavelength: reflectance_spectra[:, i] for i, wavelength in enumerate(wavelengths)}
    
    return MultiSpectralDistributions(data).align(INTERP_CMF_RANGE)


def load_msds_nikon():
    ssf = MSDS_CAMERA_SENSITIVITIES[NIKON].copy().align(INTERP_CMF_RANGE)
    return ssf

def load_msds_sigma():
    ssf = MSDS_CAMERA_SENSITIVITIES[SIGMA].align(INTERP_CMF_RANGE)
    return ssf
    
def load_msds_canon():
    ssf = sds_and_msds_to_msds(
        read_sds_from_csv_file(CANON_PATH).values()
    ).copy().align(INTERP_CMF_RANGE)
    return ssf
    
def load_cmfs():
    cmfs = MSDS_CMFS_STANDARD_OBSERVER['CIE 1931 2 Degree Standard Observer'].copy().align(INTERP_CMF_RANGE)
    illuminant = SDS_ILLUMINANTS[ILLUMINANT].align(INTERP_CMF_RANGE)
    return cmfs, illuminant
    
def load_camera(camera):
    if camera == 'nikon':
        return load_msds_nikon()
    elif camera == 'canon':
        return load_msds_canon()
    elif camera== 'sigma':
        return load_msds_sigma()
    else:
        print("Invalid camera option, valid options are: nikon, canon, sigma")
        