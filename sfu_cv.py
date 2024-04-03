from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from data import load_camera, load_cmfs, load_dataset_sfu, msds_to_rgb, msds_to_xyz
from models import GAMOptimizer, PolynomialTransformer, DeltaEOptimizer, RGBtoXYZNetwork
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
import numpy as np
RANDOM_STATE = 0

SFU_FILE_PATH = 'data/reflect_db.reflect'  # SFU material reflectance database
CAMERA = 'nikon'


# Load the SFU dataset
sfu_dataset = load_dataset_sfu(SFU_FILE_PATH)
MSDS_TRAIN = load_camera(CAMERA)

cmfs, illuminant = load_cmfs()
X = msds_to_xyz(sfu_dataset, cmfs, illuminant)

y = msds_to_rgb(sfu_dataset,MSDS_TRAIN, illuminant)
# Initialize KFold with 5 splits
kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# Define models to evaluate
models = {
    "GAMOptimizer": GAMOptimizer(lams=1e-6, order=3, n_splines=10),  # Adjust parameters as necessary
    "Linear": LinearRegression(fit_intercept=False),
    # Define other models similarly...
}

# Dictionary to hold the results
results = {model_name: [] for model_name in models}

# Cross-validation loop
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Train and evaluate each model
    for name, model in models.items():

        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Evaluate the model (using an appropriate metric, e.g., RMSE)
        score = mean_squared_error(y_test, y_pred, squared=False)

        # Append the score
        results[name].append(score)

# Calculate average scores across all folds
average_scores = {model_name: np.mean(scores) for model_name, scores in results.items()}

# Print or return the average scores
print(average_scores)
