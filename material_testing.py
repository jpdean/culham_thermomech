import numpy as np
import matplotlib.pyplot as plt

x = np.array([293.15, 323.15, 373.15, 423.15, 473.15, 523.15, 573.15, 623.15,
              673.15, 723.15, 773.15, 823.15, 873.15, 923.15, 973.15, 1023.15,
              1073.15, 1123.15, 1173.15, 1223.15, 1273.15])
y = np.array([401.0, 398.0, 395.0, 391.0, 388.0, 384.0, 381.0, 378.0, 374.0,
              371.0, 367.0, 364.0, 360.0, 357.0, 354.0, 350.0, 347.0, 344.0,
              340.0, 337.0, 334.0])


fig, ax = plt.subplots()
ax.plot(x, y)

degree = 1
coeffs = np.polynomial.Polynomial.fit(x, y, degree).convert().coef
y_poly = np.zeros_like(x)
for n in range(degree + 1):
    y_poly += coeffs[n] * x**n
ax.plot(x, y_poly)

plt.show()
