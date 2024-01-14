import numpy as np
from preprocessing import TensorBSplineTransformer, DeltaEOptimizer, PolynomialTransformer
import colour
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.metrics import make_scorer
from utils import compute_response, parse_reflectance_spectra, deltae_mean, interleave_arrays
# from patsy import dmatrix
from pygam import LinearGAM, te
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet

SFU_FILE_PATH = 'reflect_db.reflect'

    
def compute_error_sfu():
    np.random.seed(42)
    # Load spectral sensitivities
    cmfs = colour.colorimetry.MSDS_CMFS_STANDARD_OBSERVER['CIE 1931 2 Degree Standard Observer'].align(colour.SpectralShape(400, 700, 10))
    ss = colour.MSDS_CAMERA_SENSITIVITIES['Nikon 5100 (NPL)'].align(colour.SpectralShape(400, 700, 10))
        
    reflectance_spectra, _ = parse_reflectance_spectra(SFU_FILE_PATH)
    
    test_sfu = colour.MultiSpectralDistributions(reflectance_spectra).align(colour.SpectralShape(400, 700, 10))
    

    macbeth = colour.MultiSpectralDistributions(colour.SDS_COLOURCHECKERS['babel_average']).align(colour.SpectralShape(400, 700, 10))    
    # Load reference illuminant
    D65 = colour.SDS_ILLUMINANTS["D65"].align(colour.SpectralShape(400, 700, 10))
    response_human_macbeth =  compute_response(macbeth, D65, cmfs)
    response_human_sfu = compute_response(test_sfu, D65, cmfs)
    
    response_sensor_macbeth = compute_response(macbeth, D65, ss, wb=True)
    response_sensor_sfu  = compute_response(test_sfu, D65, ss, wb=True)
        
    X_train, X_test, y_train, y_test = train_test_split(response_sensor_sfu, response_human_sfu, test_size=0.2, random_state=42)
    bspline_pipeline =  Pipeline([
        ('spline_transformer', TensorBSplineTransformer(2, 3)),
        ('regressor', Ridge())
    ])
    
    start_exponent = -1  # For 1e-01
    end_exponent = -7   # For 1e-05
    
    values = [10 ** i for i in range(start_exponent, end_exponent - 1, -1)]

    
    # Define the parameter grid to search
    param_grid = {
        'spline_transformer__degree': [2,4, 5,7],
        'spline_transformer__n_knots': [3, 6, 8,],
        'regressor__alpha': values  # Regularization strength
        # Add more parameters here if you want
    }
    
    custom_scorer = make_scorer(deltae_mean, greater_is_better=False)

    

    # Set up the grid search with cross-validation
    grid_search = GridSearchCV(bspline_pipeline, param_grid, cv=5, verbose=1, n_jobs=-1, scoring=custom_scorer)

    grid_search.fit(X_train, y_train)

    pred(grid_search, X_test, y_test, "DeltaE GAM, SFU")
    pred(grid_search,response_sensor_macbeth, response_human_macbeth, "DeltaE GAM, MACBETH")
        
    macbeth_out_srgb = colour.XYZ_to_sRGB(response_human_macbeth)
    response_sensor_macbeth =  colour.XYZ_to_sRGB(grid_search.predict(response_sensor_macbeth))
    
    interleaved_array = interleave_arrays(macbeth_out_srgb, response_sensor_macbeth)

    colour.plotting.plot_multi_colour_swatches(interleaved_array, columns=6, direction="-y", spacing=0.05, legend=False, show=False, compare_swatches="Diagonal")
    plt.show(block=True)
    
    # Best parameters found
    print("Best parameters: ", grid_search.best_params_)

    # Evaluate the model on the test set
    #test_score = grid_search.score(X_test, y_test)
    #print("Test set score: ", test_score)

    

if __name__ == "__main__":
    compute_error_sfu()
