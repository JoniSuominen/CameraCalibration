import numpy as np
from colour import XYZ_to_Lab, delta_E, XYZ_to_xy, SDS_ILLUMINANTS, SpectralShape
from colour.plotting import plot_chromaticity_diagram_CIE1931
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import colour 
import torch

def degrees(n): return n * (180. / np.pi)
def radians(n): return n * (np.pi / 180.)
def hpf_diff(x, y):
    mask1=((x == 0) * (y == 0)).float()
    mask1_no = 1-mask1

    tmphp = degrees(torch.atan2(x*mask1_no, y*mask1_no))
    tmphp1 = tmphp * (tmphp >= 0).float()
    tmphp2 = (360+tmphp)* (tmphp < 0).float()

    return tmphp1+tmphp2

def dhpf_diff(c1, c2, h1p, h2p):

    mask1  = ((c1 * c2) == 0).float()
    mask1_no  = 1-mask1
    res1=(h2p - h1p)*mask1_no*(torch.abs(h2p - h1p) <= 180).float()
    res2 = ((h2p - h1p)- 360) * ((h2p - h1p) > 180).float()*mask1_no
    res3 = ((h2p - h1p)+360) * ((h2p - h1p) < -180).float()*mask1_no

    return res1+res2+res3

def ahpf_diff(c1, c2, h1p, h2p):

    mask1=((c1 * c2) == 0).float()
    mask1_no=1-mask1
    mask2=(torch.abs(h2p - h1p) <= 180).float()
    mask2_no=1-mask2
    mask3=(torch.abs(h2p + h1p) < 360).float()
    mask3_no=1-mask3

    res1 = (h1p + h2p) *mask1_no * mask2
    res2 = (h1p + h2p + 360.) * mask1_no * mask2_no * mask3 
    res3 = (h1p + h2p - 360.) * mask1_no * mask2_no * mask3_no
    res = (res1+res2+res3)+(res1+res2+res3)*mask1
    return res*0.5

def xyz_to_lab(xyz):
    # Define constants for the transformation
    epsilon = 0.008856
    kappa = 903.3
    X_n = 0.95047
    Y_n = 1.00000
    Z_n = 1.08883

    # Normalizing by the reference white point
    xyz_scaled = xyz / torch.tensor([X_n, Y_n, Z_n])

    # Calculate f(t) for XYZ
    f_xyz = torch.where(xyz_scaled > epsilon, xyz_scaled.pow(1/3), (kappa * xyz_scaled + 16) / 116)

    # Calculate Lab components
    L = 116 * f_xyz[:, 1] - 16
    a = 500 * (f_xyz[:, 0] - f_xyz[:, 1])
    b = 200 * (f_xyz[:, 1] - f_xyz[:, 2])

    # Combine L, a, b
    lab = torch.stack([L, a, b], dim=1)

    return lab


def ciede2000_diff(xyz1, xyz2):
    '''
    CIEDE2000 metric to claculate the color distance map for a batch of image tensors defined in CIELAB space
    
    '''
    Lab1 = xyz_to_lab(xyz1)
    Lab2 = xyz_to_lab(xyz2)
    
    L1 = Lab1[:, 0]
    A1 = Lab1[:,1]
    B1 = Lab1[:, 2]
    
    L2 = Lab2[:, 0]
    A2 = Lab2[:,1]
    B2 = Lab2[:, 2]
    
    kL = 1
    kC = 1
    kH = 1
    
    mask_value_0_input1=((A1==0)*(B1==0)).float()
    mask_value_0_input2=((A2==0)*(B2==0)).float()
    mask_value_0_input1_no=1-mask_value_0_input1
    mask_value_0_input2_no=1-mask_value_0_input2
    B1=B1+0.0001*mask_value_0_input1
    B2=B2+0.0001*mask_value_0_input2 
    
    C1 = torch.sqrt((A1 ** 2.) + (B1 ** 2.))
    C2 = torch.sqrt((A2 ** 2.) + (B2 ** 2.))   
   
    aC1C2 = (C1 + C2) / 2.
    G = 0.5 * (1. - torch.sqrt((aC1C2 ** 7.) / ((aC1C2 ** 7.) + (25 ** 7.))))
    a1P = (1. + G) * A1
    a2P = (1. + G) * A2
    c1P = torch.sqrt((a1P ** 2.) + (B1 ** 2.))
    c2P = torch.sqrt((a2P ** 2.) + (B2 ** 2.))


    h1P = hpf_diff(B1, a1P)
    h2P = hpf_diff(B2, a2P)
    h1P=h1P*mask_value_0_input1_no
    h2P=h2P*mask_value_0_input2_no 
    
    dLP = L2 - L1
    dCP = c2P - c1P
    dhP = dhpf_diff(C1, C2, h1P, h2P)
    dHP = 2. * torch.sqrt(c1P * c2P) * torch.sin(radians(dhP) / 2.)
    mask_0_no=1-torch.max(mask_value_0_input1,mask_value_0_input2)
    dHP=dHP*mask_0_no

    aL = (L1 + L2) / 2.
    aCP = (c1P + c2P) / 2.
    aHP = ahpf_diff(C1, C2, h1P, h2P)
    T = 1. - 0.17 * torch.cos(radians(aHP - 39)) + 0.24 * torch.cos(radians(2. * aHP)) + 0.32 * torch.cos(radians(3. * aHP + 6.)) - 0.2 * torch.cos(radians(4. * aHP - 63.))
    dRO = 30. * torch.exp(-1. * (((aHP - 275.) / 25.) ** 2.))
    rC = torch.sqrt((aCP ** 7.) / ((aCP ** 7.) + (25. ** 7.)))    
    sL = 1. + ((0.015 * ((aL - 50.) ** 2.)) / torch.sqrt(20. + ((aL - 50.) ** 2.)))
    
    sC = 1. + 0.045 * aCP
    sH = 1. + 0.015 * aCP * T
    rT = -2. * rC * torch.sin(radians(2. * dRO))

#     res_square=((dLP / (sL * kL)) ** 2.) + ((dCP / (sC * kC)) ** 2.) + ((dHP / (sH * kH)) ** 2.) + rT * (dCP / (sC * kC)) * (dHP / (sH * kH))

    res_square=((dLP / (sL * kL)) ** 2.) + ((dCP / (sC * kC)) ** 2.)*mask_0_no + ((dHP / (sH * kH)) ** 2.)*mask_0_no + rT * (dCP / (sC * kC)) * (dHP / (sH * kH))*mask_0_no
    mask_0=(res_square<=0).float()
    mask_0_no=1-mask_0
    res_square=res_square+0.0001*mask_0    
    res=torch.sqrt(res_square)
    res=res*mask_0_no
    return torch.mean(res)

def deltae_stats_nm(coef, X, y, degree, RP, n_terms):

    M = coef.reshape((3, n_terms))
    result = colour.apply_matrix_colour_correction(
            X,
            M,
            method="Finlayson 2015",
            degree=degree,
            root_polynomial_expansion=RP,
    )
    
    deltae = delta_E(XYZ_to_Lab(result), y)
    
    error = np.mean(deltae)
    return error

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
