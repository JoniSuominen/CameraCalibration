import numpy as np
from preprocessing import TensorBSplineTransformer, DeltaEOptimizer, PolynomialTransformer
import colour
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from utils import compute_response, deltae_stats, parse_reflectance_spectra, deltae_mean, interleave_arrays
import pandas as pd
# from patsy import dmatrix
from pygam import LinearGAM, te
from sklearn.linear_model import LinearRegression

SFU_FILE_PATH = 'reflect_db.reflect'

def evaluate(results, model):
    print(f"---- RESULTS {model} ----")
    print(f"DeltaE mean: {np.mean(results)}")
    print(f"DeltaE max: {np.max(results)}")
    print(f"DeltaE min: {np.min(results)}")
    print(f"DeltaE median: {np.median(results)}")
    print(f"DeltaE 95 percentile: {np.quantile(results, 0.95)}")
    print(f"DeltaE 99 percentile: {np.quantile(results, 0.99)}")
    
def k_fold_gam_cv(X, y, param_grid, n_splits=5, random_state=None):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    best_score = float('inf')
    best_lam = None
    best_order = None
    for order in param_grid['order']:
        for params in param_grid['lam']:
            cv_scores = []
            try:
                for train_idx, val_idx in kf.split(X, y):
                    print(f"Fold: {str(train_idx)}, lam: {str(params)}, order: {str(order)}")
                    predictor_X = LinearGAM(terms=te(0,1, spline_order=order) + te(0,2, spline_order=order) + te(1,2, spline_order=order) + te(0,1,2, spline_order=order), lam=params)
                    predictor_Y = LinearGAM(terms=te(0,1, spline_order=order) + te(0,2, spline_order=order) + te(1,2, spline_order=order) + te(0,1,2, spline_order=order), lam=params)
                    predictor_Z = LinearGAM(terms=te(0,1, spline_order=order) + te(0,2, spline_order=order) + te(1,2, spline_order=order) + te(0,1,2, spline_order=order), lam=params)
                    X_train, X_val = X[train_idx], X[val_idx]
                    y_train, y_val = y[train_idx], y[val_idx]
                    
                    predictor_X.fit(X_train, y_train[:, 0])
                    predictor_Y.fit(X_train, y_train[:,1])
                    predictor_Z.fit(X_train, y_train[:,2])
                    
                    Xp = predictor_X.predict(X_val)
                    Yp = predictor_Y.predict(X_val)
                    Zp = predictor_Z.predict(X_val)
                    XYZ = np.vstack((Xp,Yp,Zp)).T
                    score = deltae_mean(XYZ, y_val)
                    cv_scores.append(score)
                
                mean_score = np.mean(cv_scores)
        
                if mean_score < best_score:
                    best_score = mean_score
                    best_order = order
                    best_lam = params
            except:
                print("failed to converge")
            
    return best_lam,best_order, best_score    

def pred(model, X, y, identifier):
    XYZ = model.predict(X)
    print(XYZ.shape)
    deltae = deltae_stats(XYZ, y)
    evaluate(deltae, identifier)
    
def compute_error_sfu():
    np.random.seed(42)
    # Load spectral sensitivities
    cmfs = colour.colorimetry.MSDS_CMFS_STANDARD_OBSERVER['CIE 1931 2 Degree Standard Observer'].align(colour.SpectralShape(400, 700, 10))
    ss = colour.MSDS_CAMERA_SENSITIVITIES['Nikon 5100 (NPL)'].align(colour.SpectralShape(400, 700, 10))
        
    reflectance_spectra, _ = parse_reflectance_spectra(SFU_FILE_PATH)
    
    test_sfu = colour.MultiSpectralDistributions(reflectance_spectra).align(colour.SpectralShape(400, 700, 10))
    

    macbeth = colour.MultiSpectralDistributions(colour.SDS_COLOURCHECKERS['babel_average']).align(colour.SpectralShape(400, 700, 10))
    sd = macbeth.to_sds()
    
    # Load reference illuminant
    D65 = colour.SDS_ILLUMINANTS["D65"].align(colour.SpectralShape(400, 700, 10))
    response_human_macbeth =  compute_response(macbeth, D65, cmfs)
    response_human_sfu = compute_response(test_sfu, D65, cmfs)
    
    response_sensor_macbeth = compute_response(macbeth, D65, ss, wb=True)
    response_sensor_sfu  = compute_response(test_sfu, D65, ss, wb=True)
        
    X_train, X_test, y_train, y_test = train_test_split(response_sensor_sfu, response_human_sfu, test_size=0.2, random_state=42)
    bspline_pipeline_lab = Pipeline([
        ('spline_transformer',TensorBSplineTransformer(4, 5)),
        ('regressor', DeltaEOptimizer())
    ])
    
    bspline_pipeline_lab.fit(X_train, y_train)

    pred(bspline_pipeline_lab, X_test, y_test, "DeltaE GAM, SFU")
    pred(bspline_pipeline_lab,response_sensor_macbeth, response_human_macbeth, "DeltaE GAM, MACBETH")
        
    macbeth_out_srgb = colour.XYZ_to_sRGB(response_human_macbeth)
    response_sensor_macbeth =  colour.XYZ_to_sRGB(bspline_pipeline_lab.predict(response_sensor_macbeth))
    interleaved_array = interleave_arrays(macbeth_out_srgb, response_sensor_macbeth)

    colour.plotting.plot_multi_colour_swatches(interleaved_array, columns=6, direction="-y", spacing=0.05, legend=False, show=False, compare_swatches="Diagonal")
    plt.show(block=True)

    

if __name__ == "__main__":
    compute_error_sfu()
