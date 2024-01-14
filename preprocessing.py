from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin 
from colour import polynomial_expansion
from sklearn.preprocessing import SplineTransformer, PolynomialFeatures
from sklearn.linear_model import SGDRegressor
import numpy as np
from scipy.optimize import minimize
from utils import deltae_stats_nm, callback_function, deltae_stats_smooth, deltae_mean, optimisation_factory
from pygam import LinearGAM, te, s, l
import time
from sklearn.linear_model import Ridge
from pygam.utils import tensor_product, b_spline_basis, gen_edge_knots
import nlopt
import matplotlib.pyplot as plt
from colour.characterisation import apply_matrix_colour_correction
from colour import XYZ_to_Lab
class PolynomialTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, degree=2, rp=True, method="Finlayson 2015"):
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
        knots_double = np.tile(np.linspace(0, 1, 2*n_knots).reshape(-1, 1), (1, 3))
        knots = np.tile(np.linspace(0, 1, n_knots).reshape(-1, 1), (1, 3))
        self.cubic_transformer = SplineTransformer(degree=self.degree, knots=knots)
        self.linear_transformer = SplineTransformer(degree=1, knots=knots)
        self.knot_interval = None
        
    def fit(self, X, y=None):


        self.cubic_transformer.fit(X)
        self.linear_transformer.fit(X)
        return self
        

    def _plot_basis_functions(self):
        # Generate sample points
        x = np.linspace(0, 1, 500).reshape(-1, 1)
        X_sample = np.tile(x, (1, 3))

        # Evaluate basis functions
        cubic_spline_basis = self.cubic_transformer.transform(X_sample)
        linear_spline_basis = self.linear_transformer.transform(X_sample)

        # Plot cubic basis functions
        plt.figure(figsize=(12, 6))
        for i in range(cubic_spline_basis.shape[1]):
            plt.plot(x, cubic_spline_basis[:, i], label=f'Cubic Spline {i+1}')
        plt.title('Cubic Basis Functions')
        plt.xlabel('X')
        plt.ylabel('Basis Function Value')
        plt.legend()
        plt.show(block=True)

        # Plot linear basis functions
        plt.figure(figsize=(12, 6))
        for i in range(linear_spline_basis.shape[1]):
            plt.plot(x, linear_spline_basis[:, i], label=f'Linear Spline {i+1}')
        plt.title('Linear Basis Functions')
        plt.xlabel('X')
        plt.ylabel('Basis Function Value')
        plt.legend()
        plt.show(block=True) 


    def transform(self, X):
        poly = PolynomialFeatures(2)
        cubic_spline = self.cubic_transformer.transform(X)
        linear_spline = self.linear_transformer.transform(X)
        # linear_spline = self.cubic_transformer.transform(X[:, 0].reshape(-1, 1))
        # RG_tensor = tensor_product(R_spline, G_spline, reshape=True)  # This will be of shape (n, m*m)

        # Then, compute the tensor product of the result with B
        # RGB_tensor = tensor_product(RG_tensor, B_spline, reshape=True)        
                
        #tensor_product_der = [np.outer(r, np.outer(g, b)).flatten() for r, g, b in zip(R_der, G_der, B_der)]
        #poly_features = polynomial_expansion(X, method="Finlayson 2015", degree=2, root_polynomial_expansion=True)
        #print(np.hstack((poly_features, cubic_spline)).shape)
        return cubic_spline
    
    
class DeltaEOptimizer(BaseEstimator, RegressorMixin):
    def __init__(self, solver="BFGS", l=0.01, coefs=None):
        self.ccm = None
        self.l = l
        self.solver = solver
        self.coefs = None
        
        
    def fit(self, X, y=None):
        shape = X.shape
        self.coefs = np.ones((3 *  13))
        result = minimize(deltae_stats_nm, self.coefs, method=self.solver, args=(X, XYZ_to_Lab(y)), options={'maxiter': 2000000, 'adaptive': True})
        self.ccm = result.x.reshape((3, 13))
        return self
        
    def predict(self, X):
        return apply_matrix_colour_correction(
            X,
            self.ccm,
            method="Finlayson 2015",
            degree=3,
            root_polynomial_expansion=True,
        )
    
class GAMOptimizer(BaseEstimator, RegressorMixin):
    def __init__(self, lams, order, n_splines=10, penalties="derivative"):
        self.lams = lams
        self.order = order
        self.n_splines = n_splines
        self.penalties = penalties
        self.predictor_X = None
        self.predictor_Y = None
        self.predictor_Z = None
        
        
    def fit(self, X, y=None):
        #a = s(1, 20,2)
        #a.compile(X)
        #tee = a.build_columns(X)
        term_rg = te(0,1, spline_order=self.order, n_splines=self.n_splines, penalties=self.penalties, edge_knots=[[0, 1], [0, 1]])
        term_rb = te(0,2, spline_order=self.order, n_splines=self.n_splines, penalties=self.penalties, edge_knots=[[0, 1], [0, 1]])
        term_gb = te(1,2, spline_order=self.order, n_splines=self.n_splines, penalties=self.penalties, edge_knots=[[0, 1], [0, 1]])
        interactions = term_rg + term_rb + term_gb
        self.predictor_X = LinearGAM(terms=interactions, lam=self.lams, fit_intercept=False)
        self.predictor_Y = LinearGAM(terms=interactions, lam=self.lams, fit_intercept=False)
        self.predictor_Z = LinearGAM(terms=interactions, lam=self.lams, fit_intercept=False)
        
        #self.predictor_X = LinearGAM(terms=s(0,spline_order=5) +s(1,spline_order=5) + s(2,spline_order=5) + s(0,spline_order=3, n_splines=40) + s(1,spline_order=3, n_splines=40) + s(2,spline_order=3, n_splines=40) + te(0,1,2, spline_order=5), lam=self.lams)
        #self.predictor_Y = LinearGAM(terms=s(0,spline_order=5) +s(1,spline_order=5) + s(2,spline_order=5) + s(0,spline_order=3, n_splines=40) + s(1,spline_order=3, n_splines=40) + s(2,spline_order=3, n_splines=40) + te(0,1,2, spline_order=5), lam=self.lams)
        #self.predictor_Z = LinearGAM(terms=s(0,spline_order=5) +s(1,spline_order=5) + s(2,spline_order=5) + s(0,spline_order=3, n_splines=40) + s(1,spline_order=3, n_splines=40) + s(2,spline_order=3, n_splines=40) + te(0,1,2, spline_order=5), lam=self.lams)
        self.predictor_X.fit(X, y[:, 0])
        self.predictor_Y.fit(X, y[:, 1])
        self.predictor_Z.fit(X, y[:, 2])

        return self
    
    def plot_partial_dependences(self):
        n_features = self.predictor_X.statistics_['m_features']
        fig = plt.figure()
        for i, term in enumerate(self.predictor_X.terms):
            pos1 = i*3 + 1
            pos2 = i*3 + 2
            pos3 = i*3 + 3
            # Create subplots
            ax1 = fig.add_subplot(3, 3, pos1, projection='3d')
            ax2 = fig.add_subplot(3, 3, pos2, projection='3d')
            ax3 = fig.add_subplot(3, 3, pos3, projection='3d')
            XX = self.predictor_X.generate_X_grid(term=i, meshgrid=True)
            X = self.predictor_X.partial_dependence(term=i, X=XX, meshgrid=True)
            Y = self.predictor_Y.partial_dependence(term=i, X=XX, meshgrid=True)
            Z = self.predictor_Z.partial_dependence(term=i, X=XX, meshgrid=True)
            ax1.plot_surface(XX[0], XX[1], X, cmap='viridis')
            ax2.plot_surface(XX[0], XX[1], Y, cmap='viridis')
            ax3.plot_surface(XX[0], XX[1], Z, cmap='viridis')
        fig.tight_layout()
        plt.show(block=True)
        
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
        opt = nlopt.opt(nlopt.GN_CRS2_LM, 3 * shape[0])

        # Set optimization parameters
        opt.set_min_objective(objective_function)
        opt.set_maxtime(200)

        lower_bounds = [-3.0] * (3 * shape[0])
        upper_bounds = [3.0] * (3 * shape[0])
        opt.set_lower_bounds(lower_bounds)
        opt.set_upper_bounds(upper_bounds)

        # Perform the optimization
        coefs = opt.optimize(np.random.rand(3 * shape[0]))
        result = opt.last_optimize_result()
        print(result)
        opt2 = nlopt.opt(nlopt.LN_COBYLA, 3 * shape[0])
        opt2.set_min_objective(objective_function)
        opt2.set_maxtime(3600)
        coefs = opt2.optimize(coefs)
        self.ccm = coefs.reshape((3, shape[0]))
        return self

    def predict(self, X):
        pred = self.ccm @ X.T
        return pred.T

class TensorBSplineBasis(BaseEstimator, TransformerMixin):
    def __init__(self, degree, n_splines):
        self.n_splines = n_splines
        n_knots = n_splines - degree + 1
        self.degree = degree
        self.n_knots = n_knots
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        edge_knots = gen_edge_knots(X, dtype="numerical")
        R = b_spline_basis(X[:, 0], edge_knots, n_splines=10)
        G = b_spline_basis(X[:, 1], edge_knots, n_splines=10)
        B = b_spline_basis(X[:, 2], edge_knots, n_splines=10)
        RG_tensor = tensor_product(R, G, reshape=True)  # This will be of shape (n, m*m)

        # Then, compute the tensor product of the result with B
        RGB_tensor = tensor_product(RG_tensor, B, reshape=True)   

        print(RGB_tensor[0, 0])     
                
        #tensor_product_der = [np.outer(r, np.outer(g, b)).flatten() for r, g, b in zip(R_der, G_der, B_der)]
        
        return RGB_tensor


class SGDXYZRegressor(BaseEstimator, RegressorMixin):
    def __init__(self):
        self.X_regressor = SGDRegressor(max_iter=10000, alpha=0.1, eta0=1e-6)
        self.Y_regressor = SGDRegressor(max_iter=10000, alpha=0.1, eta0=1e-6)
        self.Z_regressor = SGDRegressor(max_iter=10000, alpha=0.1, eta0=1e-6)
        
    def fit(self, X, y=None):
        self.X_regressor.fit(X, y[:, 0])
        self.Y_regressor.fit(X, y[:, 1])
        self.Z_regressor.fit(X, y[:, 2])
        return self
        
    def predict(self, X):
        
        Xp = self.X_regressor.predict(X)
        Yp = self.Y_regressor.predict(X)
        Zp = self.Z_regressor.predict(X)
        return np.vstack((Xp,Yp,Zp)).T
    
class ColourOptimizer(BaseEstimator, RegressorMixin):
    def __init__(self):
        self.M = None

    def fit(self, X, y=None):
        (
            x_0,
            objective_function,
            xyz_to_optimization_colour_model,
            finaliser_function,
        ) = optimisation_factory()
        print(x_0)
        optimisation_settings = {
            "method": "BFGS",
        }
        res =  minimize(
            objective_function,
            x_0,
            (X, y),
            **optimisation_settings,
        )
        self.M = finaliser_function(res.x)
        
        return self
    
    def predict(self, X):
        xyz_PCC = apply_matrix_colour_correction(
            X,
            self.M,
            method="Finlayson 2015",
            degree=3,
            root_polynomial_expansion=True,
        )
        return xyz_PCC