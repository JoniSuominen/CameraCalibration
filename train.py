from pygam import LinearGAM, te
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
import numpy as np
from utils import deltae_m, deldeltae_mean


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