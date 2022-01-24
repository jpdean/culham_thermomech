import numpy as np
import matplotlib.pyplot as plt


def plot_material(x, y, coeffs):
    fig, ax = plt.subplots()
    ax.plot(x, y)
    y_poly = np.zeros_like(x)
    for n in range(len(coeffs)):
        y_poly += coeffs[n] * x**n
    ax.plot(x, y_poly)
    plt.show()


x = np.array([293.15, 323.15, 373.15, 423.15, 473.15, 523.15, 573.15, 623.15, 673.15,
             723.15, 773.15, 823.15, 873.15, 923.15, 973.15, 1023.15, 1073.15, 1123.15, 1173.15])
y = np.array([16.7, 17.0, 17.2, 17.5, 17.7, 17.8, 18.0, 18.1, 18.2,
             18.4, 18.5, 18.65, 18.8, 18.97, 19.14, 19.34, 19.55, 19.78, 20.05])

degree = 3
coeffs = np.polynomial.Polynomial.fit(x, y, degree).convert().coef

plot_material(x, y, coeffs)
