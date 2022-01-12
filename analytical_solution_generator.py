from sympy import symbols
from sympy.vector import CoordSys3D, laplacian, gradient, divergence
from sympy import sin, cos, pi, simplify

R = CoordSys3D("R")
t = symbols("t")

T = sin(pi * R.x) * cos(pi * R.y) * sin(pi * t)

print(simplify(T.diff(t) - laplacian(T)))
