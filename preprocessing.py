from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin 
from colour import polynomial_expansion
from sklearn.preprocessing import SplineTransformer
import numpy as np
from scipy.optimize import minimize
from utils import deltae_stats_nm, callback_function
from pygam import LinearGAM, te, s
import time
from sklearn.linear_model import Ridge
from pygam.utils import tensor_product

class PolynomialTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, degree=3, rp=True, method="Finlayson 2015"):
        self.degree = degree
        self.rp = rp
        self.method = method

    def fit(self, X, y=None):
        # Determine the knot locations based on the training data
        return self

    def transform(self, X):
        # Create the cubic spline terms using the knot locations
        return polynomial_expansion(X, method=self.method, degree=self.degree, root_polynomial_expansion=self.rp)


class TensorBSplineTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, degree=2, n_knots=20):
        self.degree = degree
        self.n_knots = n_knots
        self.r_transformer = None
        self.g_transformer = None
        self.b_transformer = None
        #self.r_derivative  = None
        #self.g_derivative  = None
        #self.b_derivative = None
        self.knot_interval = None
        
    def fit(self, X, y=None):
        self.r_transformer = SplineTransformer(degree=self.degree, n_knots=self.n_knots, extrapolation="linear")
        self.g_transformer = SplineTransformer(degree=self.degree, n_knots=self.n_knots, extrapolation="linear")
        self.b_transformer = SplineTransformer(degree=self.degree, n_knots=self.n_knots, extrapolation="linear")
        self.r_transformer.fit(X[:,0].reshape(-1, 1))
        self.g_transformer.fit(X[:,1].reshape(-1, 1))
        self.b_transformer.fit(X[:,2].reshape(-1, 1))
        #self.knot_interval = abs(self.r_transformer.bsplines_[0].t[1]-self.r_transformer.bsplines_[0].t[0])
        # self.derivative_knots = 
        
        #r_knots = self.r_transformer.bsplines_[0].t[self.degree:-self.degree] - (self.knot_interval / 2)
        #g_knots = self.g_transformer.bsplines_[0].t[self.degree:-self.degree] - (self.knot_interval / 2)
        #b_knots = self.b_transformer.bsplines_[0].t[self.degree:-self.degree] - (self.knot_interval / 2)
        #self.r_derivative = SplineTransformer(degree=self.degree - 2, knots = r_knots.reshape(-1, 1) , extrapolation="linear")
        #self.g_derivative = SplineTransformer(degree=self.degree - 2, knots = g_knots.reshape(-1, 1) , extrapolation="linear")
        #self.b_derivative = SplineTransformer(degree=self.degree - 2, knots = b_knots.reshape(-1, 1) , extrapolation="linear")
        #self.r_derivative.fit(X[:,0].reshape(-1, 1))
        #self.g_derivative.fit(X[:,1].reshape(-1, 1))
        #self.b_derivative.fit(X[:,2].reshape(-1, 1))
        return self
        
    def transform(self, X):
        R = self.r_transformer.transform(X[:, 0].reshape(-1, 1))
        G = self.g_transformer.transform(X[:, 1].reshape(-1, 1))
        B = self.b_transformer.transform(X[:, 2].reshape(-1, 1))
        #R_der = self.r_derivative.transform(X[:, 0].reshape(-1, 1))
        #G_der = self.g_derivative.transform(X[:, 1].reshape(-1, 1))
        #B_der = self.g_derivative.transform(X[:, 2].reshape(-1, 1))

        #tensor_product_rg = [np.outer(r, g).flatten() for r, g in zip(R, G)]
        #tensor_product_gb = [np.outer(g, b).flatten() for g, b in zip(G, B)]
        #tensor_product_rb = [np.outer(r, b).flatten() for r, b in zip(R, B)]
        # tensor_product_rgb = [np.outer(r, np.outer(g, b)).flatten() for r, g, b in zip(R, G, B)]
        RG_tensor = tensor_product(B, G, reshape=True)  # This will be of shape (n, m*m)

        # Then, compute the tensor product of the result with B
        RGB_tensor = tensor_product(R, RG_tensor, reshape=True)
        
        #tensor_product_der = [np.outer(r, np.outer(g, b)).flatten() for r, g, b in zip(R_der, G_der, B_der)]
        
        return RGB_tensor
    
    
class DeltaEOptimizer(BaseEstimator, RegressorMixin):
    def __init__(self, solver="BFGS"):
        self.ccm = None
        self.solver = solver
        
        
    def fit(self, X, y=None):
        shape = X[0, :].shape
        result = minimize(deltae_stats_nm, np.random.rand(3*shape[0]), method=self.solver, args=(X, y, (3, shape[0])),  options={'maxiter': 20000000})
        self.ccm = result.x.reshape((3, shape[0]))
        return self
        
    def predict(self, X):
        pred = self.ccm @ X.T
        return pred.T
    
class GAMOptimizer(BaseEstimator, RegressorMixin):
    def __init__(self, lams):
        self.lams = lams
        self.predictor_X = None
        self.predictor_Y = None
        self.predictor_Z = None
        
        
    def fit(self, X, y=None):
        order = 2
        #a = s(1, 20,2)
        #a.compile(X)
        #tee = a.build_columns(X)
        self.predictor_X = LinearGAM(terms=te(0,1,2, spline_order=order, penalties="l2"), lam=self.lams)
        self.predictor_Y = LinearGAM(terms=te(0,1,2, spline_order=order, penalties="l2"), lam=self.lams)
        self.predictor_Z = LinearGAM(terms=te(0,1,2, spline_order=order, penalties="l2"), lam=self.lams)
        
        #self.predictor_X = LinearGAM(terms=s(0,spline_order=5) +s(1,spline_order=5) + s(2,spline_order=5) + s(0,spline_order=3, n_splines=40) + s(1,spline_order=3, n_splines=40) + s(2,spline_order=3, n_splines=40) + te(0,1,2, spline_order=5), lam=self.lams)
        #self.predictor_Y = LinearGAM(terms=s(0,spline_order=5) +s(1,spline_order=5) + s(2,spline_order=5) + s(0,spline_order=3, n_splines=40) + s(1,spline_order=3, n_splines=40) + s(2,spline_order=3, n_splines=40) + te(0,1,2, spline_order=5), lam=self.lams)
        #self.predictor_Z = LinearGAM(terms=s(0,spline_order=5) +s(1,spline_order=5) + s(2,spline_order=5) + s(0,spline_order=3, n_splines=40) + s(1,spline_order=3, n_splines=40) + s(2,spline_order=3, n_splines=40) + te(0,1,2, spline_order=5), lam=self.lams)
        self.predictor_X.fit(X, y[:, 0])
        self.predictor_Y.fit(X, y[:, 1])
        self.predictor_Z.fit(X, y[:, 2])
        
        print(self.predictor_X.summary())    
        return self
        
    def predict(self, X):
        Xp = self.predictor_X.predict(X)
        Yp = self.predictor_Y.predict(X)
        Zp = self.predictor_Z.predict(X)
        return np.vstack((Xp,Yp,Zp)).T