import numpy as np
from colour import XYZ_to_Lab, delta_E, XYZ_to_xy, SDS_ILLUMINANTS, SpectralShape
from colour.plotting import plot_chromaticity_diagram_CIE1931
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

def compute_response(reflectance,  illuminant, sensitivities, wb=False):
    response = reflectance.values.T @ np.diag(illuminant.values) @ sensitivities.values
    max_response = compute_response_perfect_reflector(sensitivities)
    if wb:
        response /= max_response
    else:
        response /= max_response[1]
    return response


def compute_response_perfect_reflector(sensitivities):
    illuminant = SDS_ILLUMINANTS["D65"].align(SpectralShape(400, 700, 10))
    response_test = np.diag(illuminant.values) @ sensitivities.values
    response_sum = np.sum(response_test, axis=0)
    
    return response_sum
def wb(response_macbeth, responses):
    
    r_Gain = response_macbeth[18, 0] / response_macbeth[18, 1]
    b_Gain = response_macbeth[18, 2] / response_macbeth[18, 1]
    
    gains = [r_Gain, 1, b_Gain]
    
    response_sensor = responses / gains
    return response_sensor

def parse_reflectance_spectra(file_path):
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
    
    return data, wavelengths

def callback_function(current_solution):
    print("Current solution:", current_solution)

def deltae_stats_nm(coef, a, b, c, lambda_l1=0.005):
    ccm = coef.reshape(c)
    result = ccm @ a.T
    deltae = delta_E(XYZ_to_Lab(result.T), XYZ_to_Lab(b))
    
    #penalty = 0.00005 * np.sum(np.abs(np.diff(coef, n=2, axis=0)))
    l1_penalty = lambda_l1 * np.sum(coef**2)
    return np.mean(deltae) + l1_penalty

def deltae_stats(a,b):
    sensor_lab = XYZ_to_Lab(a)
    human_lab = XYZ_to_Lab(b)
    
    deltae = delta_E(sensor_lab, human_lab)
    return deltae 

def deltae_mean(a,b):
    sensor_lab = XYZ_to_Lab(b)
    human_lab = XYZ_to_Lab(a)

    deltae = delta_E(sensor_lab, human_lab)
    return np.mean(deltae)

def deltae_stats_cross(estimator,a,b):
    sensor_response_estimator = estimator.predict(a)
    sensor_response_estimator = sensor_response_estimator / np.max(sensor_response_estimator)
    sensor_lab = XYZ_to_Lab(sensor_response_estimator)
    human_lab = XYZ_to_Lab(b)
    
    deltae = delta_E(human_lab, sensor_lab)
    return np.max(deltae)

def plot_chromaticity_diagram(responses_xyz):
    responses_xy = XYZ_to_xy(responses_xyz)
    plot_chromaticity_diagram_CIE1931(standalone=False)    

    plt.plot(responses_xy[:, 0], responses_xy[:, 1], 'bo', markersize=8)
    
    plt.plot([0.64,0.3, 0.15, 0.64],
        [0.33,0.60, 0.06, 0.33],
        color='white', linewidth=2)
    
    plt.show(block=True)
    
def interleave_arrays(array1, array2):
    # This function assumes both arrays have the same length
    interleaved = []
    for a, b in zip(array1, array2):
        interleaved.extend([a, b])
    return interleaved