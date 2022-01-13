from sympy import symbols
from sympy.vector import CoordSys3D, gradient, divergence
from sympy import sin, cos, pi, simplify

R = CoordSys3D("R")
t = symbols("t")

T = sin(pi * R.x) * cos(pi * R.y) * sin(pi * t)
kappa = 4.1 + T**2
c = 1.3 + T**2
rho = 2.7 + T**2

f = simplify(rho * c * T.diff(t) - divergence(kappa * gradient(T)))

# FIXME This currently adds undesired spacing to powers
replacement_strings = (("pi", "np.pi"),
                       ("sin", "np.sin"),
                       ("cos", "np.cos"),
                       ("R.x", "x[0]"),
                       ("R.y", "x[1]"),
                       ("*", " * "))
f_code = str(f)
for s in replacement_strings:
    f_code = f_code.replace(s[0], s[1])

print(f_code)
