from sympy import symbols
from sympy.vector import CoordSys3D, gradient, divergence
from sympy import sin, cos, pi, simplify

R = CoordSys3D("R")
t, c, rho, kappa = symbols("t c rho kappa")

T = sin(pi * R.x) * cos(pi * R.y) * sin(pi * t)

print(simplify(rho * c * T.diff(t) - divergence(kappa * gradient(T))))
