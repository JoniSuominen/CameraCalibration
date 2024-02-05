import numpy as np
from colour_math import deltae_stats


def evaluate(results, model):
    print(f"---- RESULTS {model} ----")
    print(f"DeltaE mean: {np.mean(results)}")
    print(f"DeltaE max: {np.max(results)}")
    print(f"DeltaE min: {np.min(results)}")
    print(f"DeltaE median: {np.median(results)}")
    print(f"DeltaE 95 percentile: {np.quantile(results, 0.95)}")
    print(f"DeltaE 99 percentile: {np.quantile(results, 0.99)}")
  

def pred(model, X, y, identifier):
    XYZ = model.predict(X)
    XYZ2 = XYZ / np.max(y)
    Y2 = y / np.max(y)
    deltae = deltae_stats(XYZ2, Y2)
    evaluate(deltae, identifier)