from materials import materials as mat_dict
import gmsh
from mpi4py import MPI
from dolfinx.io import gmshio
from dolfinx import mesh, fem
import numpy as np
from petsc4py import PETSc
from thermomech import solve


def create_mesh(comm, h):
    volume_ids = {"volume": 1}
    boundary_ids = {"front": 1,
                    "back": 2,
                    "sides": 3}

    gmsh.initialize()
    if comm.rank == 0:
        gmsh.model.add("domain")

        r_o = 0.5
        r_i = 0.25
        outer = gmsh.model.occ.addCylinder(0, 0, 0, 2, 0, 0, r_o)
        inner = gmsh.model.occ.addCylinder(0, 0, 0, 2, 0, 0, r_i)
        gmsh.model.occ.cut([(3, outer)], [(3, inner)], 3)

        gmsh.model.occ.synchronize()

        gmsh.model.addPhysicalGroup(3, [3], volume_ids["volume"])

        gmsh.model.addPhysicalGroup(2, [3], boundary_ids["back"])
        gmsh.model.addPhysicalGroup(2, [2], boundary_ids["front"])
        gmsh.model.addPhysicalGroup(2, [1, 4], boundary_ids["sides"])

        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), h)
        gmsh.model.mesh.generate(3)

        # gmsh.write("test_mesh.msh")
        # exit()

    partitioner = mesh.create_cell_partitioner(mesh.GhostMode.none)
    msh, ct, ft = gmshio.model_to_mesh(
        gmsh.model, comm, 0, gdim=3, partitioner=partitioner)
    gmsh.finalize()

    return msh, ct, ft, volume_ids, boundary_ids


comm = MPI.COMM_WORLD
h = 0.1
msh, ct, ft, volume_ids, boundary_ids = create_mesh(comm, h)

delta_t = 10
num_time_steps = 25
# Polynomial order
k = 1

# Materials
materials = {volume_ids["volume"]: mat_dict["CuCrZr"]}

# Specify boundary conditions
bcs = {}
bcs["T"] = {boundary_ids["back"]:
            {"type": "temperature",
             "value": lambda x: 293.15 * np.ones_like(x[0])},
            boundary_ids["sides"]:
            {"type": "convection",
                "value": lambda x: 293.15 * np.ones_like(x[0]),
                "h": lambda T: 5},
            boundary_ids["front"]:
            {"type": "heat_flux",
             "value": lambda x: 1e5 * np.ones_like(x[0])}}
bcs["u"] = {boundary_ids["back"]:
            {"type": "displacement",
                "value": np.array([0, 0, 0], dtype=PETSc.ScalarType)}}

bc_mt = {}
bc_mt["T"] = ft
bc_mt["u"] = ft

# Elastic source function (not including gravity)
f_u = fem.Constant(msh, np.array([0, 0, 0], dtype=PETSc.ScalarType))


# Thermal source function
def f_T(x): return np.zeros_like(x[0])


# Initial temperature
def T_0(x): return 293.15 * np.ones_like(x[0])


# Acceleration due to gravity
g = PETSc.ScalarType(- 9.81)

# Solve the problem
results = solve(msh, k, delta_t, num_time_steps, T_0, f_T,
                f_u, g, materials, ct, bcs, bc_mt,
                write_to_file=True, steps_per_write=1)
