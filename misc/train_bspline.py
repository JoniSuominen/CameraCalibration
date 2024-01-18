# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
# ---

from ast import Del
from evaluate import pred
from preprocessing import ColourOptimizer, DeltaEOptimizer, NLOptOptimizer, PolynomialTransformer, GAMOptimizer
import numpy as np
import colour
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from utils import compute_response, deltae_mean, parse_reflectance_spectra, interleave_arrays
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.experimental import enable_halving_search_cv # noqa
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.metrics import make_scorer
from skopt import BayesSearchCV
from skopt.space import Real, Integer
np.int = np.int64
import pandas as pd
from sklearn.inspection import partial_dependence, PartialDependenceDisplay

# Constants
SFU_FILE_PATH = 'reflect_db.reflect'
CMF_RANGE = colour.SpectralShape(400, 700, 10)
ILLUMINANT = "D65"
CAMERA = 'Nikon 5100 (NPL)'
TEST_SIZE = 0.1
RANDOM_STATE = 0
np.random.seed(RANDOM_STATE)
def load_spectral_data(csv_file_path):
    """
    Load spectral data from a CSV file.
    The first column is assumed to be non-spectral (e.g., an identifier).
    """
    df = pd.read_csv(csv_file_path)
    spectral_data = df.iloc[:, 1:].to_numpy()
    return spectral_data.T


def load_data():
    """
    Load and align spectral data.
    """
    cmfs = colour.colorimetry.MSDS_CMFS_STANDARD_OBSERVER['CIE 1931 2 Degree Standard Observer'].align(CMF_RANGE)
    ss = colour.MSDS_CAMERA_SENSITIVITIES[CAMERA].align(CMF_RANGE)
    
    train_set, _ = parse_reflectance_spectra(SFU_FILE_PATH)
    test_set = load_spectral_data("foster_new.csv")
    test_data = colour.MultiSpectralDistributions(test_set, CMF_RANGE)
    train_data = colour.MultiSpectralDistributions(train_set).align(CMF_RANGE)
    macbeth = colour.MultiSpectralDistributions(colour.SDS_COLOURCHECKERS['babel_average'])
    D65 = colour.SDS_ILLUMINANTS[ILLUMINANT].align(CMF_RANGE)

    return cmfs, ss, train_data, test_data, D65

def compute_and_split_responses(cmfs, ss, train, test, D65):
    """
    Compute responses and split data for training and testing.
    """
    norm_d65 = colour.characterisation.normalise_illuminant(D65, ss)

    response_testset_xyz = colour.characterisation.training_data_sds_to_XYZ(test, cmfs, D65, None)
    response_trainset_xyz = colour.characterisation.training_data_sds_to_XYZ(train, cmfs, D65, None)
    response_testset_sensor, _ = colour.characterisation.training_data_sds_to_RGB(test, ss, norm_d65)
    response_trainset_sensor, _  = colour.characterisation.training_data_sds_to_RGB(train, ss, norm_d65)

    
    x_train, x_test, y_train, y_test = train_test_split(response_trainset_sensor, response_trainset_xyz, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    return (x_train, x_test, y_train, y_test), response_testset_sensor, response_testset_xyz

def train_model(X_train, y_train):
    """
    Train the model with the given data.
    """
    
    custom_scorer = make_scorer(deltae_mean, greater_is_better=False)
    model = GAMOptimizer(lams=0.0001, order=3, n_splines=20)
    
    RP_linear = Pipeline([
        ('regressor', DeltaEOptimizer())
    ])
    
    
    param_grid = {
        'regressor__l': [1*10**(-i) for i in range(1,)],  # Example values, adjust as needed
        'spline_transformer__degree': [1,2,3],  # Example values, adjust as needed
        'spline_transformer__n_splines': [3,4,5],  # Example values, adjust as needed
    }

    param_grid_gam = {
        'lams': [1*10**(-i) for i in range(2, 10)],
    }
    
    
    bayes_search = BayesSearchCV(
        model,
        param_grid_gam,
        n_iter=50,  # Number of iterations for the Bayesian optimization
        cv=5,
        n_jobs=-1,
        scoring=custom_scorer,
        verbose=2,
        random_state=RANDOM_STATE,  # Set a random seed for reproducibility
    )


    model.fit(X_train, y_train)
    #features = [0, 1, 2, (0, 1), (0, 2), (1, 2)]
    #feature_range = np.linspace(0, 1, 50)
    #xx0, xx1, xx2 = np.meshgrid(feature_range, feature_range, feature_range, indexing='ij')

    # Now, flatten the grid to feed into the regressor
    #grid = np.vstack([xx0.ravel(), xx1.ravel(), xx2.ravel()]).T
    
    #PartialDependenceDisplay.from_estimator(RP_linear, grid, features, target=0)
    #plt.show(block=True)
    
    # print("Best parameters:", bayes_search.best_params_)
    model.plot_partial_dependences()
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
    pred(model, X_test, y_test, "DeltaE Foster+CAVE")
    pred(model, response_sensor_macbeth, response_human_macbeth, "DeltaE SFU")
    # plot_results(model, response_sensor_macbeth, response_human_macbeth)

if __name__ == "__main__":
    main()
