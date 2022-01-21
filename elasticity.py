import numpy as np

from dolfinx.fem import (VectorFunctionSpace, dirichletbc,
                         locate_dofs_topological, FunctionSpace, Function,
                         Constant, form, assemble_matrix, assemble_vector,
                         apply_lifting, set_bc)
from dolfinx.io import XDMFFile
from dolfinx.mesh import create_unit_square
import ufl

from mpi4py import MPI
from petsc4py import PETSc
from problems import create_mesh_tags


def sigma(v, T, T_ref, alpha_L, E, nu):
    # Elastic strain
    eps_e = ufl.sym(ufl.grad(v))
    eps_T = alpha_L * (T - T_ref) * ufl.Identity(len(v))
    eps = eps_e - eps_T
    mu = E / (2.0 * (1.0 + nu))
    lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return 2.0 * mu * eps + lmbda * ufl.tr(eps) * ufl.Identity(len(v))


def solve(mesh, k, T, f, materials, material_mt, bcs, bc_mt):
    V = VectorFunctionSpace(mesh, ("Lagrange", k))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    F = - ufl.inner(f, v) * ufl.dx

    dx = ufl.Measure("dx", domain=mesh, subdomain_data=material_mt)

    for marker, mat in enumerate(materials):
        (alpha_L, T_ref) = mat["thermal_strain"]
        E = mat["E"]
        nu = mat["nu"]
        F += ufl.inner(
            sigma(u, T, T_ref, alpha_L(T), E(T), nu), ufl.grad(v)) * dx(marker)

    ds = ufl.Measure("ds", domain=mesh, subdomain_data=bc_mt)

    dirichlet_bcs = []
    for marker, bc in enumerate(bcs):
        bc_type = bc["type"]
        if bc_type == "dirichlet":
            facets = np.array(
                bc_mt.indices[bc_mt.values == marker])
            dofs = locate_dofs_topological(V, bc_mt.dim, facets)
            dirichlet_bcs.append(
                dirichletbc(bc["value"], dofs, V))
        elif bc_type == "pressure":
            F -= ufl.inner(bc["value"] * ufl.FacetNormal(mesh), v) * ds(marker)
        else:
            raise Exception(
                f"Boundary condition type {bc_type} not recognised")

    a = form(ufl.lhs(F))
    L = form(ufl.rhs(F))

    A = assemble_matrix(a, bcs=dirichlet_bcs)
    A.assemble()

    b = assemble_vector(L)
    apply_lifting(b, [a], bcs=[dirichlet_bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b, dirichlet_bcs)

    ksp = PETSc.KSP().create(MPI.COMM_WORLD)

    # Set matrix operator
    ksp.setOperators(A)
    ksp.setType(PETSc.KSP.Type.PREONLY)
    ksp.getPC().setType(PETSc.PC.Type.LU)

    # Compute solution
    # ksp.setMonitor(
    #     lambda ksp, its, rnorm:
    #     print("Iteration: {}, rel. residual: {}".format(its, rnorm)))
    u_h = Function(V)
    u_h.name = "u"
    ksp.solve(b, u_h.vector)
    # ksp.view()

    with XDMFFile(MPI.COMM_WORLD, "u.xdmf", "w") as file:
        file.write_mesh(mesh)
        file.write_function(u_h)

    return u_h


def main():
    n = 32
    k = 1
    mesh = create_unit_square(MPI.COMM_WORLD, n, n)

    V = FunctionSpace(mesh, ("Lagrange", k))
    T = Function(V)
    T.interpolate(lambda x: np.sin(np.pi * x[0]) * np.cos(np.pi * x[1]) + 1)

    materials = []
    materials.append({"name": "mat_1",
                      "nu": 0.33,
                      "E": lambda T: 1.0 + 0.1 * T**2,
                      "thermal_strain": (lambda T: 0.1 + 0.01 * T**3, 1.5)})
    materials.append({"name": "mat_2",
                      "nu": 0.1,
                      "E": lambda T: 1.0 + 0.5 * T**2,
                      "thermal_strain": (lambda T: 0.2 + 0.015 * T**2, 1.0)})
    materials_mt = create_mesh_tags(
        mesh,
        [lambda x: x[0] <= 0.5, lambda x: x[0] >= 0.5],
        mesh.topology.dim)

    bcs = [{"type": "dirichlet",
            "value": np.array([0, 0], dtype=PETSc.ScalarType)},
           {"type": "pressure",
            "value": Constant(mesh, PETSc.ScalarType(-1))}]
    bc_mt = create_mesh_tags(
        mesh,
        [lambda x: np.logical_or(np.logical_or(np.isclose(x[0], 0.0),
                                               np.isclose(x[0], 1.0)),
                                 np.isclose(x[1], 0.0)),
         lambda x: np.isclose(x[1], 1.0)],
        mesh.topology.dim - 1)

    f = Constant(mesh, np.array([0, -1], dtype=PETSc.ScalarType))

    solve(mesh, k, T, f, materials, materials_mt, bcs, bc_mt)


if __name__ == "__main__":
    main()
