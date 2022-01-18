import numpy as np

from dolfinx.fem import (VectorFunctionSpace, dirichletbc,
                         locate_dofs_geometrical, LinearProblem)
from dolfinx.io import XDMFFile
from dolfinx.mesh import create_unit_square
from ufl import (Identity, TestFunction, TrialFunction, as_vector,
                 dx, grad, inner, sym, tr)

from mpi4py import MPI
from petsc4py import PETSc


n = 32
mesh = create_unit_square(MPI.COMM_WORLD, n, n)

# Young's modulus
E = 1.0e9
# Poisson's ratio
nu = 0.33

mu = E / (2.0 * (1.0 + nu))
lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))


def sigma(v):
    # Elastic strain
    eps_e = sym(grad(v))
    eps = eps_e  # TODO Add thermal strain
    return 2.0 * mu * eps + lmbda * tr(eps) * Identity(
        len(v))


V = VectorFunctionSpace(mesh, ("Lagrange", 1))

u = TrialFunction(V)
v = TestFunction(V)
f = as_vector((0.0, - 1.0))
a = inner(sigma(u), grad(v)) * dx
L = inner(f, v) * dx  # TODO Add pressure BC

bc = dirichletbc(np.array([0, 0, 0], dtype=PETSc.ScalarType),
                 locate_dofs_geometrical(V, lambda x: np.isclose(x[0], 0.0)),
                 V)

problem = LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly",
                                                       "pc_type": "lu"})
u = problem.solve()

with XDMFFile(MPI.COMM_WORLD, "u.xdmf", "w") as file:
    file.write_mesh(mesh)
    file.write_function(u)
