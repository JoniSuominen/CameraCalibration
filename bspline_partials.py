from evaluate import pred
from preprocessing import TensorBSplineTransformer, DeltaEOptimizer, NLOptOptimizer, PolynomialTransformer, GAMOptimizer
import numpy as np
import colour
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from utils import compute_response, deltae_mean, parse_reflectance_spectra, interleave_arrays
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.experimental import enable_halving_search_cv # noqa
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.metrics import make_scorer
from skopt import BayesSearchCV
from skopt.space import Real, Integer
np.int = np.int64
import pandas as pd

# Constants
SFU_FILE_PATH = 'reflect_db.reflect'
CMF_RANGE = colour.SpectralShape(400, 700, 10)
ILLUMINANT = "D65"
CAMERA = 'Nikon 5100 (NPL)'
TEST_SIZE = 0.2
RANDOM_STATE = 0
np.random.seed(RANDOM_STATE)
def load_spectral_data(csv_file_path):
    """
    Load spectral data from a CSV file.
    The first column is assumed to be non-spectral (e.g., an identifier).
    """
    df = pd.read_csv(csv_file_path)
    spectral_data = df.iloc[:, 1:].to_numpy()
    return spectral_data


def load_data():
    """
    Load and align spectral data.
    """
    cmfs = colour.colorimetry.MSDS_CMFS_STANDARD_OBSERVER['CIE 1931 2 Degree Standard Observer'].align(CMF_RANGE)
    ss = colour.MSDS_CAMERA_SENSITIVITIES[CAMERA].align(CMF_RANGE)
    
    reflectance_spectra, _ = parse_reflectance_spectra(SFU_FILE_PATH)
    foster_spectra = load_spectral_data("combined.csv")
    print(foster_spectra.shape)
    test_foster = colour.MultiSpectralDistributions(foster_spectra.T, CMF_RANGE).align(CMF_RANGE)
    test_sfu = colour.MultiSpectralDistributions(reflectance_spectra).align(CMF_RANGE)
    macbeth = colour.MultiSpectralDistributions(colour.SDS_COLOURCHECKERS['babel_average']).align(CMF_RANGE)
    D65 = colour.SDS_ILLUMINANTS[ILLUMINANT].align(CMF_RANGE)

    return cmfs, ss, test_foster, macbeth, D65

def compute_and_split_responses(cmfs, ss, test_sfu, macbeth, D65):
    """
    Compute responses and split data for training and testing.
    """
    norm_d65 = colour.characterisation.normalise_illuminant(D65, ss)

    response_human_macbeth = colour.characterisation.training_data_sds_to_XYZ(macbeth, cmfs, D65, None)
    response_human_sfu = colour.characterisation.training_data_sds_to_XYZ(test_sfu, cmfs, D65, None)
    response_sensor_macbeth, wb_macbeth = colour.characterisation.training_data_sds_to_RGB(macbeth, ss, norm_d65)
    response_sensor_sfu, wb_sfu = colour.characterisation.training_data_sds_to_RGB(test_sfu, ss, norm_d65)
    
    x_train, x_test, y_train, y_test = train_test_split(response_sensor_sfu, response_human_sfu, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    #n_samples = 200
    #n_train_samples = x_train.shape[0]
    #indices = np.random.choice(n_train_samples, n_samples, replace=False)
    
    #print(indices)



    return (x_train, x_test, y_train, y_test), response_sensor_macbeth, response_human_macbeth

def train_model(X_train, y_train):
    """
    Train the model with the given data.
    """
    
    custom_scorer = make_scorer(deltae_mean, greater_is_better=False)
    model = GAMOptimizer(lams=0.001, order=3, n_splines=10)
    # model.fit(X_train, y_train)
    #coefs = np.concatenate((model.predictor_X.coef_, model.predictor_Y.coef_, model.predictor_Z.coef_))

    model.fit(X_train, y_train)


    # print("Best parameters:", bayes_search.best_params_)
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
    print(X_train.shape)
    n_samples = X_train.shape[0]
    print(n_samples)
    model = train_model(X_train[0:n_samples], y_train[0:n_samples])
    #ccm = model.named_steps['regressor'].ccm
    #print(np.max(ccm))
    pred(model, X_test, y_test, "DeltaE GAM, SFU")
    pred(model, response_sensor_macbeth, response_human_macbeth, "DeltaE GAM, MACBETH")
    # plot_results(model, response_sensor_macbeth, response_human_macbeth)

if __name__ == "__main__":
    main()
