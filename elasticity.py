import numpy as np

from dolfinx.fem import (VectorFunctionSpace, dirichletbc,
                         locate_dofs_geometrical, LinearProblem,
                         assemble_scalar, form, FunctionSpace,
                         Function)
from dolfinx.io import XDMFFile
from dolfinx.mesh import create_unit_square
import ufl

from mpi4py import MPI
from petsc4py import PETSc
from problems import create_problem_0


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
    # FIXME Should be p * n, not p \cdot n
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

    problem = create_problem_0(mesh)

    u_e = problem["u"]

    # Compute solution at t = 1.5
    T_expr = problem["T"]
    T_expr.t = 1.5
    V = FunctionSpace(mesh, ("Lagrange", k))
    T = Function(V)
    T.interpolate(T_expr)

    material = problem["materials"][0]
    (alpha_L, T_ref) = material["thermal_strain"]
    nu = material["nu"]
    E = material["E"]

    f = - ufl.div(sigma(u_e, T, T_ref, alpha_L(T), E(T), nu))
    # FIXME Make this a scalar
    p = sigma(u_e, T, T_ref, alpha_L(T), E(T), nu)

    u_h = solve(mesh, k, nu, T, T_ref, E(T), alpha_L(T), f, p)

    error_L2 = np.sqrt(mesh.comm.allreduce(assemble_scalar(
        form((u_h - u_e)**2 * ufl.dx)), op=MPI.SUM))
    if mesh.comm.Get_rank() == 0:
        print(error_L2)


if __name__ == "__main__":
    main()
