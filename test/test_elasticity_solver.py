import ufl
from problems import (create_mesh_tags, compute_error_L2_norm,
                      compute_convergence_rate)
import numpy as np

from dolfinx.fem import FunctionSpace, Function
from dolfinx.mesh import create_unit_square

from mpi4py import MPI
from petsc4py import PETSc

import elasticity


def alpha_L(T): return 0.1 + 0.01 * T**3
def E(T): return 1.0 + 0.1 * T**2


n = 32
k = 1
T_ref = 1.5
nu = 0.33
materials = []
materials.append({"name": "mat_1",
                  "nu": nu,
                  "E": E,
                  "thermal_strain": (alpha_L, T_ref)})
materials.append({"name": "mat_2",
                  "nu": nu,
                  "E": E,
                  "thermal_strain": (alpha_L, T_ref)})

mesh = create_unit_square(MPI.COMM_WORLD, n, n)

materials_mt = create_mesh_tags(
    mesh,
    [lambda x: x[0] <= 0.5, lambda x: x[0] >= 0.5],
    mesh.topology.dim)


V = FunctionSpace(mesh, ("Lagrange", k))
T = Function(V)
T.interpolate(lambda x: np.sin(np.pi * x[0]) * np.cos(np.pi * x[1]) + 1)

x = ufl.SpatialCoordinate(mesh)
u = ufl.as_vector((ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1]),
                   ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])))

f = - ufl.div(elasticity.sigma(u, T, T_ref, alpha_L(T), E(T), nu))
# NOTE p here is a rank 2 tensor for convenience, rather than
# a scalar
p = elasticity.sigma(u, T, T_ref, alpha_L(T), E(T), nu)

bcs = [{"type": "dirichlet",
        "value": np.array([0, 0], dtype=PETSc.ScalarType)},
       {"type": "pressure",
        "value": p}]
bc_mt = create_mesh_tags(
    mesh,
    [lambda x: np.logical_or(np.logical_or(np.isclose(x[0], 0.0),
                                           np.isclose(x[0], 1.0)),
                             np.isclose(x[1], 0.0)),
        lambda x: np.isclose(x[1], 1.0)],
    mesh.topology.dim - 1)


u_h = elasticity.solve(mesh, k, T, f, materials, materials_mt, bcs, bc_mt)

print(compute_error_L2_norm(mesh.comm, u_h, u))
