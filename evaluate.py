import numpy as np
from colour_math import deltae_stats


def evaluate(results, model):
    print(f"---- RESULTS {model} ----")
    print(f"DeltaE mean: {np.mean(results):.2f}")
    print(f"DeltaE max: {np.max(results):.2f}")
    print(f"DeltaE median: {np.median(results):.2f}")
    print(f"DeltaE 95 percentile: {np.quantile(results, 0.95):.2f}")
    print(f"DeltaE 99 percentile: {np.quantile(results, 0.99):.2f}")
  

def pred(model, X, y, identifier):
    XYZ = model.predict(X)
    deltae = deltae_stats(XYZ, y)
    evaluate(deltae, identifier)