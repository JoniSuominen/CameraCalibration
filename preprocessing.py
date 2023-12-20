from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin 
from colour import polynomial_expansion
from sklearn.preprocessing import SplineTransformer, PolynomialFeatures
import numpy as np
from scipy.optimize import minimize
from utils import deltae_stats_nm, callback_function, deltae_stats_smooth
from pygam import LinearGAM, te, s
import time
from sklearn.linear_model import Ridge
from pygam.utils import tensor_product
import nlopt

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
    def __init__(self, degree, n_splines):
        self.n_splines = n_splines
        n_knots = n_splines - degree + 1
        self.degree = degree
        self.n_knots = n_knots
        self.r_transformer = None
        self.g_transformer = None
        self.b_transformer = None
        # self.interaction_transformer = PolynomialFeatures(degree=3, interaction_only=True)
        #self.r_derivative  = None
        #self.g_derivative  = None
        #self.b_derivative = None
        self.knot_interval = None
        
    def fit(self, X, y=None):
        self.r_transformer = SplineTransformer(degree=self.degree, n_knots=self.n_knots)
        self.g_transformer = SplineTransformer(degree=self.degree, n_knots=self.n_knots)
        self.b_transformer = SplineTransformer(degree=self.degree, n_knots=self.n_knots)
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
        R_spline = self.r_transformer.transform(X[:, 0].reshape(-1, 1))
        G_spline = self.g_transformer.transform(X[:, 1].reshape(-1, 1))
        B_spline = self.b_transformer.transform(X[:, 2].reshape(-1, 1))
        RG_tensor = tensor_product(R_spline, G_spline, reshape=True)  # This will be of shape (n, m*m)

        # Then, compute the tensor product of the result with B
        RGB_tensor = tensor_product(RG_tensor, B_spline, reshape=True)        
                
        #tensor_product_der = [np.outer(r, np.outer(g, b)).flatten() for r, g, b in zip(R_der, G_der, B_der)]
        
        return RGB_tensor
    
    
class DeltaEOptimizer(BaseEstimator, RegressorMixin):
    def __init__(self, solver="BFGS", l=0.01):
        self.ccm = None
        self.l = l
        self.solver = solver
        
        
    def fit(self, X, y=None):
        shape = X[0, :].shape
        result = minimize(deltae_stats_smooth, np.random.rand(3*shape[0]), method=self.solver, args=(X, y, (3, shape[0]), self.l), options={'maxiter': 2000000, 'adaptive': True})
        print(result)
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
        order = 7
        #a = s(1, 20,2)
        #a.compile(X)
        #tee = a.build_columns(X)
        self.predictor_X = LinearGAM(terms= te(0,1,2, spline_order=order) , lam=self.lams)
        self.predictor_Y = LinearGAM(terms= te(0,1,2, spline_order=order), lam=self.lams)
        self.predictor_Z = LinearGAM(terms= te(0,1,2, spline_order=order), lam=self.lams)
        
        #self.predictor_X = LinearGAM(terms=s(0,spline_order=5) +s(1,spline_order=5) + s(2,spline_order=5) + s(0,spline_order=3, n_splines=40) + s(1,spline_order=3, n_splines=40) + s(2,spline_order=3, n_splines=40) + te(0,1,2, spline_order=5), lam=self.lams)
        #self.predictor_Y = LinearGAM(terms=s(0,spline_order=5) +s(1,spline_order=5) + s(2,spline_order=5) + s(0,spline_order=3, n_splines=40) + s(1,spline_order=3, n_splines=40) + s(2,spline_order=3, n_splines=40) + te(0,1,2, spline_order=5), lam=self.lams)
        #self.predictor_Z = LinearGAM(terms=s(0,spline_order=5) +s(1,spline_order=5) + s(2,spline_order=5) + s(0,spline_order=3, n_splines=40) + s(1,spline_order=3, n_splines=40) + s(2,spline_order=3, n_splines=40) + te(0,1,2, spline_order=5), lam=self.lams)
        self.predictor_X.fit(X, y[:, 0])
        self.predictor_Y.fit(X, y[:, 1])
        self.predictor_Z.fit(X, y[:, 2])
            
        print(self.predictor_X.summary())    
        print(self.predictor_Y.summary())
        print(self.predictor_Z.summary())
        return self
        
    def predict(self, X):
        Xp = self.predictor_X.predict(X)
        Yp = self.predictor_Y.predict(X)
        Zp = self.predictor_Z.predict(X)
        return np.vstack((Xp,Yp,Zp)).T
    
    
class NLOptOptimizer(BaseEstimator, RegressorMixin):
    def __init__(self, solver="UOBYQA"):
        self.ccm = None
        self.solver = solver

    def fit(self, X, y=None):
        shape = X[0, :].shape

        # Define the optimization function
        def objective_function(ccm, grad):
            if grad.size > 0:
                # Compute gradient here if needed
                pass
            return deltae_stats_nm(ccm, X, y, (3, shape[0]))

        # Initialize the optimizer
        opt = nlopt.opt(nlopt.LN_NEWUOA, 3 * shape[0])

        # Set optimization parameters
        opt.set_min_objective(objective_function)
        opt.set_maxeval(100000)

        # Perform the optimization
        self.ccm = opt.optimize(np.random.rand(3 * shape[0])).reshape((3, shape[0]))
        return self

    def predict(self, X):
        pred = self.ccm @ X.T
        return pred.T