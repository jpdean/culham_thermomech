import numpy as np

from dolfinx.fem import (VectorFunctionSpace, dirichletbc,
                         locate_dofs_geometrical, LinearProblem)
from dolfinx.io import XDMFFile
from dolfinx.mesh import create_unit_square
import ufl

from mpi4py import MPI
from petsc4py import PETSc


n = 32
mesh = create_unit_square(MPI.COMM_WORLD, n, n)

# Young's modulus
E = 1.0
# Poisson's ratio
nu = 0.33

mu = E / (2.0 * (1.0 + nu))
lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))


def sigma(v):
    # Elastic strain
    eps_e = ufl.sym(ufl.grad(v))
    eps = eps_e  # TODO Add thermal strain
    return 2.0 * mu * eps + lmbda * ufl.tr(eps) * ufl.Identity(
        len(v))


x = ufl.SpatialCoordinate(mesh)
u_e = ufl.as_vector((ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1]),
                     ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])))
f = - ufl.div(sigma(u_e))

V = VectorFunctionSpace(mesh, ("Lagrange", 1))

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
a = ufl.inner(sigma(u), ufl.grad(v)) * ufl.dx
L = ufl.inner(f, v) * ufl.dx  # TODO Add pressure BC

bc = dirichletbc(np.array([0, 0, 0], dtype=PETSc.ScalarType),
                 locate_dofs_geometrical(V, lambda x: np.logical_or(np.logical_or(np.isclose(x[0], 0.0),
                                                                                  np.isclose(x[0], 1.0)),
                                                                    np.logical_or(np.isclose(x[1], 0.0),
                                                                                  np.isclose(x[1], 1.0)))),
                 V)

problem = LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly",
                                                       "pc_type": "lu"})
u_h = problem.solve()

with XDMFFile(MPI.COMM_WORLD, "u.xdmf", "w") as file:
    file.write_mesh(mesh)
    file.write_function(u_h)
