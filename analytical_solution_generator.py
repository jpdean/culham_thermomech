from sympy import symbols
from sympy.vector import CoordSys3D, gradient, divergence
from sympy import sin, cos, pi, simplify


def sympy_to_code(sympy_func):
    replacement_strings = (("pi", "np.pi"),
                           ("sin", "np.sin"),
                           ("cos", "np.cos"),
                           ("R.x", "x[0]"),
                           ("R.y", "x[1]"),
                           ("t", "self.t"),
                           ("*", " * "))
    code = str(sympy_func)
    for s in replacement_strings:
        code = code.replace(s[0], s[1])

    # HACK To fix replaced power spacing.
    # FIXME Implement function properly with regex
    code = code.replace(" *  * ", "**")
    return code


R = CoordSys3D("R")
t = symbols("t")

T = sin(pi * R.x) * cos(pi * R.y) * sin(pi * t)
kappa = 4.1 + T**2
c = 1.3 + T**2
rho = 2.7 + T**2
# Heat transfer coefficient
h = 3.5 + T**2

f = simplify(rho * c * T.diff(t) - divergence(kappa * gradient(T)))

f_code = sympy_to_code(f)
print(f"f = {f_code}")

kappa_grad_T = simplify(kappa * gradient(T))
# Evaluate Neumann BC on right face
kappa_dT_dn_right = simplify(kappa_grad_T.dot(R.i))
kappa_dT_dn_right_code = sympy_to_code(kappa_dT_dn_right)
print(f"\ndT_dn_right = {kappa_dT_dn_right_code}")

# Solve for T_inf on top boundary
T_inf_left = simplify(T + 1 / h * kappa_grad_T.dot(- R.i))
T_inf_left_code = sympy_to_code(T_inf_left)
print(f"\nT_inf_left = {T_inf_left_code}")
