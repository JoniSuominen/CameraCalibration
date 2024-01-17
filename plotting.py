from colour import XYZ_to_xy
from colour.plotting import plot_chromaticity_diagram_CIE1931
import matplotlib.pyplot as plt

def interleave_arrays(array1, array2):
    # This function assumes both arrays have the same length
    interleaved = []
    for a, b in zip(array1, array2):
        interleaved.extend([a, b])
    return interleaved

def plot_chromaticity_diagram(responses_xyz):
    responses_xy = XYZ_to_xy(responses_xyz)
    plot_chromaticity_diagram_CIE1931(standalone=False)    

    

    plt.scatter(responses_xy[:, 0], responses_xy[:, 1], edgecolors='black', facecolors='red')
        
    plt.plot([0.64,0.3, 0.15, 0.64],
        [0.33,0.6, 0.06, 0.33],
        color='black', linewidth=2)
    
    plt.show(block=True)