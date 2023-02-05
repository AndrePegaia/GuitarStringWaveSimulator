import numpy as np
import matplotlib.pyplot as plt
y0 = np.zeros(101)
for x in range(101):
    y0[x] = -0.000004*(x**2 - 100*x +15)