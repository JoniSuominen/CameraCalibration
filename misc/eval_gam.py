from evaluate import pred
from preprocessing import GAMOptimizer, TensorBSplineTransformer, DeltaEOptimizer, NLOptOptimizer
import numpy as np
import colour
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from utils import compute_response, deltae_mean, parse_reflectance_spectra, interleave_arrays
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

# Constants
SFU_FILE_PATH = 'reflect_db.reflect'
CMF_RANGE = colour.SpectralShape(400, 700, 10)
ILLUMINANT = "D65"
CAMERA = 'Nikon 5100 (NPL)'
TEST_SIZE = 0.2
RANDOM_STATE = 42

def load_data():
    """
    Load and align spectral data.
    """
    cmfs = colour.colorimetry.MSDS_CMFS_STANDARD_OBSERVER['CIE 1931 2 Degree Standard Observer'].align(CMF_RANGE)
    ss = colour.MSDS_CAMERA_SENSITIVITIES[CAMERA].align(CMF_RANGE)
    
    reflectance_spectra, _ = parse_reflectance_spectra(SFU_FILE_PATH)
    test_sfu = colour.MultiSpectralDistributions(reflectance_spectra).align(CMF_RANGE)
    macbeth = colour.MultiSpectralDistributions(colour.SDS_COLOURCHECKERS['babel_average']).align(CMF_RANGE)
    D65 = colour.SDS_ILLUMINANTS[ILLUMINANT].align(CMF_RANGE)

    return cmfs, ss, test_sfu, macbeth, D65

def compute_and_split_responses(cmfs, ss, test_sfu, macbeth, D65):
    """
    Compute responses and split data for training and testing.
    """
    response_human_macbeth = compute_response(macbeth, D65, cmfs)
    response_human_sfu = compute_response(test_sfu, D65, cmfs)
    response_sensor_macbeth = compute_response(macbeth, D65, ss, wb=True)
    response_sensor_sfu = compute_response(test_sfu, D65, ss, wb=True)

    return train_test_split(response_sensor_sfu, response_human_sfu, test_size=TEST_SIZE, random_state=RANDOM_STATE), response_sensor_macbeth, response_human_macbeth

def train_model(X_train, y_train):
    """
    Train the model with the given data.
    """
    model = GAMOptimizer(lams=1e-2)
    model.fit(X_train, y_train)
    
    return model

def plot_results(model, response_sensor_macbeth, response_human_macbeth):
    """
    Plot the results.
    """
    macbeth_out_srgb = colour.XYZ_to_sRGB(response_human_macbeth)
    response_sensor_macbeth = colour.XYZ_to_sRGB(model.predict(response_sensor_macbeth))
    interleaved_array = interleave_arrays(macbeth_out_srgb, response_sensor_macbeth)
    colour.plotting.plot_multi_colour_swatches(interleaved_array, columns=6, direction="-y", spacing=0.05, legend=False, show=False, compare_swatches="Diagonal")
    plt.show(block=True)

def main():
    cmfs, ss, test_sfu, macbeth, D65 = load_data()
    (X_train, X_test, y_train, y_test), response_sensor_macbeth, response_human_macbeth = compute_and_split_responses(cmfs, ss, test_sfu, macbeth, D65)
    model = train_model(X_train, y_train)
    #ccm = model.named_steps['regressor'].ccm
    #print(np.max(ccm))
    pred(model, X_test, y_test, "DeltaE GAM, SFU")
    pred(model, response_sensor_macbeth, response_human_macbeth, "DeltaE GAM, MACBETH")
    plot_results(model, response_sensor_macbeth, response_human_macbeth)

if __name__ == "__main__":
    main()
