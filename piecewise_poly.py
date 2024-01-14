import numpy as np
import matplotlib.pyplot as plt

# Generate some data
x = np.linspace(0, 1, 100)
y = 0.5 * x * x + 0.5

# Fit a piecewise polynomial with discontinuities
def piecewise_polynomial_with_discontinuities(x, a0, a1, b0, b1):
    if x < 0.5:
        return a0 + a1 * x
    else:
        return b0 + b1 * x

# Fit the data
params = np.linalg.lstsq(np.vstack([x, np.ones(len(x))]).T, y, rcond=-1)[0]

# Split the parameters into the two halves
a0, a1 = params[:2]
b0, b1 = params[2:]

# Evaluate the piecewise polynomial at each data point
y_fit = piecewise_polynomial_with_discontinuities(x, a0, a1, b0, b1)

# Fit a piecewise polynomial without discontinuities
def piecewise_polynomial_without_discontinuities(x, a0, a1, a2):
    return a0 + a1 * x + a2 * x**2

params_smooth = np.linalg.lstsq(np.vstack([x, np.ones(len(x)), x**2]).T, y, rcond=-1)[0]
a0_smooth, a1_smooth, a2_smooth = params_smooth

y_fit_smooth = piecewise_polynomial_without_discontinuities(x, a0_smooth, a1_smooth, a2_smooth)

# Plot the data and the fitted polynomials
plt.figure(figsize=(10, 6))
plt.plot(x, y, label='Data')
plt.plot(x, y_fit, label='Piecewise polynomial with discontinuities', linestyle='--', color='red')
plt.plot(x, y_fit_smooth, label='Piecewise polynomial without discontinuities', color='blue')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Comparison of piecewise polynomials with and without discontinuities')
plt.show()
