import matplotlib.pyplot as plt
import numpy as np


labels = ['50', '100', '200', '400', '600', '800', '1000', '2000', '3000', '5000']
robust_inference_vals = [0.90, 0.80, 0.74, 0.677, 0.671, 0.651, 0.644, 0.6327, 0.6097, 0.6077]
linear_regression = [0.94, 0.82, 0.75, 0.681, 0.672, 0.6512, 0.644, 0.6327, 0.6097, 0.6077]
missForest = [0.9204, 0.8090, 0.7470, 0.6646, 0.6397, 0.6111, 0.5945, 0.5744, 0.5616, 0.5590]
mice = [1.46, 1.31, 1.25, 1.082288322, 1.07, 1.05, 1.03, 1.02, 1.013, 1.007]

plt.plot(labels, robust_inference_vals)
plt.plot(labels, linear_regression)
plt.plot(labels, missForest)
plt.show()
