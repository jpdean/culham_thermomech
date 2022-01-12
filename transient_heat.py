import numpy as np

from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import fem
from dolfinx.mesh import create_unit_square, locate_entities_boundary
from dolfinx.io import XDMFFile

import ufl


def solve(mesh, k, t_end, num_time_steps, T_i_expression):
    V = fem.FunctionSpace(mesh, ("Lagrange", k))

    facet_dim = mesh.topology.dim - 1
    boundary_facets = locate_entities_boundary(
        mesh, facet_dim, lambda x: np.full(
            x.shape[1], True, dtype=bool))
    bc = fem.dirichletbc(
        PETSc.ScalarType(0), fem.locate_dofs_topological(
            V, facet_dim, boundary_facets),
        V)

    # Time step
    t = 0
    delta_t = fem.Constant(mesh, PETSc.ScalarType(t_end / num_time_steps))

    xdmf_file = XDMFFile(MPI.COMM_WORLD, "T.xdmf", "w")
    xdmf_file.write_mesh(mesh)

    T_h = fem.Function(V)
    T_h.name = "T"
    T_h.interpolate(T_i_expression)
    xdmf_file.write_function(T_h, t)

    T_n = fem.Function(V)
    T_n.x.array[:] = T_h.x.array

    T = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    f = fem.Constant(mesh, PETSc.ScalarType(0))

    a = ufl.inner(T, v) * ufl.dx + \
        delta_t * ufl.inner(ufl.grad(T), ufl.grad(v)) * ufl.dx
    L = ufl.inner(T_n + delta_t * f, v) * ufl.dx
    bilinear_form = fem.form(a)
    linear_form = fem.form(L)

    A = fem.assemble_matrix(bilinear_form, bcs=[bc])
    A.assemble()
    b = fem.create_vector(linear_form)

    solver = PETSc.KSP().create(mesh.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.PREONLY)
    solver.getPC().setType(PETSc.PC.Type.LU)

    for n in range(num_time_steps):
        t += delta_t.value

        with b.localForm() as loc_b:
            loc_b.set(0)
        fem.assemble_vector(b, linear_form)

        fem.apply_lifting(b, [bilinear_form], [[bc]])
        b.ghostUpdate(
            addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        fem.set_bc(b, [bc])

        solver.solve(b, T_h.vector)
        T_h.x.scatter_forward()

        T_n.x.array[:] = T_h.x.array

        xdmf_file.write_function(T_h, t)

    xdmf_file.close()

    return T_h


t_end = 0.1
num_time_steps = 500
n = 50
k = 1
mesh = create_unit_square(MPI.COMM_WORLD, n, n)
solve(mesh, k, t_end, num_time_steps,
      lambda x: np.exp(-50 * ((x[0] - 0.5)**2 + (x[1] - 0.5)**2)))
