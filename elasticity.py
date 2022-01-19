import numpy as np

from dolfinx.fem import (VectorFunctionSpace, dirichletbc,
                         locate_dofs_geometrical, LinearProblem,
                         assemble_scalar, form)
from dolfinx.io import XDMFFile
from dolfinx.mesh import create_unit_square
import ufl

from mpi4py import MPI
from petsc4py import PETSc


def sigma(v, T, T_ref, alpha_L, E, nu):
    # Elastic strain
    eps_e = ufl.sym(ufl.grad(v))
    eps_T = alpha_L * (T - T_ref) * ufl.Identity(len(v))
    eps = eps_e - eps_T
    mu = E / (2.0 * (1.0 + nu))
    lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return 2.0 * mu * eps + lmbda * ufl.tr(eps) * ufl.Identity(len(v))


def solve(mesh, k, nu, T, T_ref, E, alpha_L, f, p):
    V = VectorFunctionSpace(mesh, ("Lagrange", k))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    F = ufl.inner(sigma(u, T, T_ref, alpha_L, E, nu), ufl.grad(v)) * ufl.dx - \
        ufl.inner(f, v) * ufl.dx - \
        ufl.inner(ufl.dot(p, ufl.FacetNormal(mesh)), v) * ufl.ds
    a = ufl.lhs(F)
    L = ufl.rhs(F)

    def dirichlet_boundary(x):
        return np.logical_or(np.logical_or(np.isclose(x[0], 0.0),
                                           np.isclose(x[0], 1.0)),
                             np.isclose(x[1], 0.0))

    bc = dirichletbc(np.array([0, 0, 0], dtype=PETSc.ScalarType),
                     locate_dofs_geometrical(V, dirichlet_boundary),
                     V)

    problem = LinearProblem(a, L, bcs=[bc],
                            petsc_options={"ksp_type": "preonly",
                                           "pc_type": "lu"})
    u_h = problem.solve()
    u_h.name = "u"

    with XDMFFile(MPI.COMM_WORLD, "u.xdmf", "w") as file:
        file.write_mesh(mesh)
        file.write_function(u_h)

    return u_h


def main():
    n = 32
    k = 1
    mesh = create_unit_square(MPI.COMM_WORLD, n, n)
    # Poisson's ratio
    nu = 0.33
    # Thermal expansion coefficient
    # FIXME Do this properly
    T_ref = 1.5
    x = ufl.SpatialCoordinate(mesh)
    T = T_ref + ufl.cos(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    # Young's modulus
    E = 1.0 + 0.1 * T**2
    alpha_L = 0.1 + 0.01 * T**3
    u_e = ufl.as_vector((ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1]),
                         ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])))
    f = - ufl.div(sigma(u_e, T, T_ref, alpha_L, E, nu))
    # Here, purely for simplicity and the purposes of comparing with the
    # analytical solution, `p` is a tensor not a scalar, since
    # \sigma \cdot n = p n. Could also find a scalar, but there isn't much
    # point in this case.
    p = sigma(u_e, T, T_ref, alpha_L, E, nu)

    u_h = solve(mesh, k, nu, T, T_ref, E, alpha_L, f, p)

    error_L2 = np.sqrt(mesh.comm.allreduce(assemble_scalar(
        form((u_h - u_e)**2 * ufl.dx)), op=MPI.SUM))
    if mesh.comm.Get_rank() == 0:
        print(error_L2)


if __name__ == "__main__":
    main()
