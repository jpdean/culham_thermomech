from materials import materials as mat_dict
from dolfinx.io import XDMFFile
from mpi4py import MPI
import numpy as np
from petsc4py import PETSc
from dolfinx import fem
from thermomech import solve

with XDMFFile(MPI.COMM_WORLD, "csut.xdmf", "r") as f:
    mesh = f.read_mesh()
    mesh.topology.create_connectivity(2, 0)
    bc_mt = {}
    bc_mt["T"] = f.read_meshtags(mesh, "boundaries_T")
    bc_mt["u"] = f.read_meshtags(mesh, "boundaries_u")
    material_mt = f.read_meshtags(mesh, "materials")

t_end = 20
k = 1
num_time_steps = 4

materials = []
materials.append(mat_dict["Copper"])
materials.append(mat_dict["CuCrZr"])
materials.append(mat_dict["304SS"])
materials.append(mat_dict["304SS"])
materials.append(mat_dict["304SS"])

bcs = {}
bcs["T"] = [{"type": "convection",
             "value": lambda x: 293.15 * np.ones_like(x[0]),
             "h": lambda T: 5},
            {"type": "heat_flux",
             "value": lambda x: 1.6e5 * np.ones_like(x[0])},
            {"type": "heat_flux",
             "value": lambda x: 5e5 * np.ones_like(x[0])},
            {"type": "convection",
             "value": lambda x: 293.15 * np.ones_like(x[0]),
             "h": mat_dict["water"]["h"]}]
bcs["u"] = [{"type": "displacement",
             "value": np.array([0, 0, 0], dtype=PETSc.ScalarType)},
            {"type": "displacement",
             "value": np.array([0, 0, 0], dtype=PETSc.ScalarType)},
            {"type": "pressure",
             "value": fem.Constant(mesh, PETSc.ScalarType(-1e3))}]

f_u = fem.Constant(mesh, np.array([0, 0, 0], dtype=PETSc.ScalarType))


def f_T(x): return np.zeros_like(x[0])


def T_i(x): return 293.15 * np.ones_like(x[0])


g = PETSc.ScalarType(- 9.81)

solve(mesh, k, t_end, num_time_steps, T_i, f_T,
      f_u, g, materials, material_mt, bcs, bc_mt,
      write_to_file=True)
