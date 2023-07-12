from materials import materials as mat_dict
from dolfinx.io import XDMFFile
from mpi4py import MPI
import numpy as np
from petsc4py import PETSc
from dolfinx import fem
from thermomech import solve
import json


# Simulation parameters
delta_t = 5
num_time_steps = 300
write_to_file = True
steps_per_write = 25  # Number of time steps per write to file
k = 1  # Polynomial order

# Load mesh and meshtags
with XDMFFile(MPI.COMM_WORLD, "csut.xdmf", "r") as f:
    mesh = f.read_mesh()
    mesh.topology.create_connectivity(2, 0)
    bc_mt = {}
    bc_mt["T"] = f.read_meshtags(mesh, "boundaries_T")
    bc_mt["u"] = f.read_meshtags(mesh, "boundaries_u")
    material_mt = f.read_meshtags(mesh, "materials")

# TODO Give meaningful volume and boundary names
volume_ids = {"volume_0": 0,
              "volume_1": 1,
              "volume_2": 2,
              "volume_3": 3,
              "volume_4": 4}
boundary_ids = {}
boundary_ids["T"] = {"boundary_0": 0,
                     "boundary_1": 1,
                     "boundary_2": 2,
                     "boundary_3": 3}
boundary_ids["u"] = {"boundary_0": 0,
                     "boundary_1": 1,
                     "boundary_1": 2}

# Add materials
materials = {volume_ids["volume_0"]: mat_dict["Copper"],
             volume_ids["volume_1"]: mat_dict["CuCrZr"],
             volume_ids["volume_2"]: mat_dict["304SS"],
             volume_ids["volume_3"]: mat_dict["304SS"],
             volume_ids["volume_4"]: mat_dict["304SS"]}


# Boundary conditions
bcs = {}
bcs["T"] = {boundary_ids["T"]["boundary_0"]: {"type": "convection",
                                              "value": lambda x: 293.15 * np.ones_like(x[0]),
                                              "h": lambda T: 5},
            boundary_ids["T"]["boundary_1"]: {"type": "heat_flux",
                                              "value": lambda x: 1.6e5 * np.ones_like(x[0])},
            boundary_ids["T"]["boundary_2"]: {"type": "heat_flux",
                                              "value": lambda x: 5e5 * np.ones_like(x[0])},
            boundary_ids["T"]["boundary_3"]: {"type": "convection",
                                              "value": lambda x: 293.15 * np.ones_like(x[0]),
                                              "h": mat_dict["water"]["h"]}}
bcs["u"] = {boundary_ids["u"]["boundary_0"]: {"type": "displacement",
                                              "value": np.array([0, 0, 0], dtype=PETSc.ScalarType)},
            boundary_ids["u"]["boundary_1"]: {"type": "displacement",
                                              "value": np.array([0, 0, 0], dtype=PETSc.ScalarType)},
            boundary_ids["u"]["boundary_2"]: {"type": "pressure",
                                              "value": fem.Constant(mesh, PETSc.ScalarType(-1e3))}}

# Elastic source term (not including gravity)
f_u = fem.Constant(mesh, np.array([0, 0, 0], dtype=PETSc.ScalarType))


# Thermal source term
def f_T(x): return np.zeros_like(x[0])


# Initial temperature
def T_0(x): return 293.15 * np.ones_like(x[0])


# Acceleration due to gravity
g = PETSc.ScalarType(- 9.81)

# Solve the problem
results = solve(mesh, k, delta_t, num_time_steps, T_0, f_T,
                f_u, g, materials, material_mt, bcs, bc_mt,
                write_to_file=write_to_file, steps_per_write=steps_per_write)

# Write timing and iteration data to file
n_procs = MPI.COMM_WORLD.Get_size()
if mesh.comm.Get_rank() == 0:
    with open(f"csut_{n_procs}.json", "w") as f:
        json.dump(results["data"], f)
