from sympy import symbols
from sympy.vector import CoordSys3D, gradient, divergence
from sympy import sin, cos, pi, simplify

R = CoordSys3D("R")
t, c, rho = symbols("t c rho")

T = sin(pi * R.x) * cos(pi * R.y) * sin(pi * t)
kappa = 4.1 + T**2

print(simplify(rho * c * T.diff(t) - divergence(kappa * gradient(T))))
