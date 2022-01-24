import numpy as np
import matplotlib.pyplot as plt


def plot_material(x, y, coeffs_list):
    fig, ax = plt.subplots()
    ax.plot(x, y)

    num_pieces = len(coeffs_list)
    x_split = np.array_split(x, num_pieces)
    for i, coeffs in enumerate(coeffs_list):
        y_poly = np.zeros_like(x_split[i])
        for n in range(len(coeffs)):
            y_poly += coeffs[n] * x_split[i]**n
        ax.plot(x_split[i], y_poly)
    plt.show()


def fit_piecewise_poly(x, y, degree, num_pieces):
    coeffs_list = []
    x_split = np.array_split(x, num_pieces)
    y_split = np.array_split(y, num_pieces)
    for i in range(num_pieces):
        coeffs_list.append(
            np.polynomial.Polynomial.fit(
                x_split[i], y_split[i], degree).convert().coef)
    return coeffs_list


x = np.array([293.15, 323.15, 373.15, 423.15, 473.15, 523.15, 573.15, 623.15, 673.15,
             723.15, 773.15, 823.15, 873.15, 923.15, 973.15, 1023.15, 1073.15, 1123.15, 1173.15])
y = np.array([16.7, 17.0, 17.2, 17.5, 17.7, 17.8, 18.0, 18.1, 18.2,
             18.4, 18.5, 18.65, 18.8, 18.97, 19.14, 19.34, 19.55, 19.78, 20.05])

degree = 1
num_splits = 3
coeffs_list = fit_piecewise_poly(x, y, degree, num_splits)

plot_material(x, y, coeffs_list)
