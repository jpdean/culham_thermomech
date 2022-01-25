from materials import materials as mat_dict
from utils import create_mesh_tags
import numpy as np
from mpi4py import MPI
from dolfinx.mesh import create_box
from petsc4py import PETSc
from dolfinx import fem
import thermomech


# FIXME Mesh does not necessarily align with materials


t_end = 10000
num_time_steps = 100
n = 16
k = 1
mesh = create_box(
    MPI.COMM_WORLD,
    [np.array([0.0, 0.0, 0.0]),
     np.array([2.0, 1.0, 1.0])],
    [n, n, n])

# TODO Let solver take dictionary of materials instead of list
materials = []
materials.append(mat_dict["Copper"])
materials.append(mat_dict["CuCrZr"])

material_mt = create_mesh_tags(
    mesh,
    [lambda x: x[0] <= 0.5,
     lambda x: x[0] >= 0.5],
    mesh.topology.dim)

bcs = [{"type": "temperature",
        "value": lambda x: 293.15 * np.ones_like(x[0])},
       {"type": "convection",
        "value": lambda x: 393.15 * np.ones_like(x[0]),
        "h": lambda T: 5},
       {"type": "displacement",
        "value": np.array([0, 0, 0], dtype=PETSc.ScalarType)},
       {"type": "pressure",
        "value": fem.Constant(mesh, PETSc.ScalarType(-1e4))}]


def convection_boundary_marker(x):
    r = np.isclose(x[0], 1.0)
    tb = np.logical_or(np.isclose(x[1], 0.0), np.isclose(x[1], 1.0))
    fb = np.logical_or(np.isclose(x[2], 0.0), np.isclose(x[2], 1.0))
    tbfb = np.logical_or(tb, fb)
    return np.logical_or(r, tbfb)


bc_mt = create_mesh_tags(
    mesh,
    [lambda x: np.isclose(x[0], 0.0),
     convection_boundary_marker,
     lambda x: np.isclose(x[0], 0),
     lambda x: np.isclose(x[1], 1.0)],
    mesh.topology.dim - 1)

f_u = fem.Constant(mesh, np.array([0, 0, 0], dtype=PETSc.ScalarType))


def f_T(x): return np.zeros_like(x[0])


def T_i(x): return 293.15 * np.ones_like(x[0])


g = PETSc.ScalarType(- 9.81)
thermomech.solve(mesh, k, t_end, num_time_steps, T_i, f_T,
                 f_u, g, materials, material_mt, bcs, bc_mt)
