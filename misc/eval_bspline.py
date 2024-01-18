from evaluate import pred
from preprocessing import TensorBSplineTransformer, DeltaEOptimizer, PolynomialTransformer
import colour
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from utils import compute_response, parse_reflectance_spectra, interleave_arrays


SFU_FILE_PATH = 'reflect_db.reflect'
    
def compute_error_sfu():
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
    bspline_pipeline_lab = Pipeline([
        ('spline_transformer',TensorBSplineTransformer(2, 7)),
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
