import numpy as np

from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import fem
from dolfinx.mesh import locate_entities_boundary
from dolfinx.io import XDMFFile

import ufl


def solve(mesh, k, t_end, num_time_steps, problem):
    V = fem.FunctionSpace(mesh, ("Lagrange", k))

    facet_dim = mesh.topology.dim - 1
    boundary_facets = locate_entities_boundary(
        mesh, facet_dim, lambda x: np.full(
            x.shape[1], True, dtype=bool))
    T_d = fem.Function(V)
    T_d.interpolate(problem.T)
    bc = fem.dirichletbc(
        T_d, fem.locate_dofs_topological(V, facet_dim, boundary_facets))

    # Time step
    delta_t = fem.Constant(mesh, PETSc.ScalarType(t_end / num_time_steps))

    xdmf_file = XDMFFile(MPI.COMM_WORLD, "T.xdmf", "w")
    xdmf_file.write_mesh(mesh)

    T_h = fem.Function(V)
    T_h.name = "T"
    T_h.interpolate(problem.T)
    xdmf_file.write_function(T_h, problem.t)

    T_n = fem.Function(V)
    T_n.x.array[:] = T_h.x.array

    T = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    f = fem.Function(V)
    f.interpolate(problem.f)

    rho = fem.Constant(mesh, PETSc.ScalarType(problem.rho()))
    c = fem.Constant(mesh, PETSc.ScalarType(problem.c()))
    kappa = fem.Constant(mesh, PETSc.ScalarType(problem.kappa()))

    a = ufl.inner(rho * c * T, v) * ufl.dx + \
        delta_t * ufl.inner(kappa * ufl.grad(T), ufl.grad(v)) * ufl.dx
    L = ufl.inner(rho * c * T_n + delta_t * f, v) * ufl.dx
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
        problem.t += delta_t.value

        T_d.interpolate(problem.T)
        f.interpolate(problem.f)

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

        xdmf_file.write_function(T_h, problem.t)

    xdmf_file.close()

    return T_h


class Problem():
    def __init__(self):
        self.t = 0

    def T(self, x):
        return np.sin(np.pi * x[0]) * np.cos(np.pi * x[1]) * \
            np.sin(np.pi * self.t)

    def f(self, x):
        c = self.c()
        rho = self.rho()
        kappa = self.kappa()
        return np.pi * np.sin(x[0] * np.pi) * np.cos(x[1] * np.pi) * \
            (2 * np.pi * kappa * np.sin(ufl.pi * self.t) +
             rho * c * np.cos(np.pi * self.t))

    # Specific heat capacity
    def c(self):
        return 1.3

    # Density
    def rho(self):
        return 2.7

    # Thermal conductivity
    def kappa(self):
        return 4.1


def main():
    t_end = 2
    num_time_steps = 100
    n = 32
    k = 1
    mesh = create_unit_square(MPI.COMM_WORLD, n, n)
    problem = Problem()
    solve(mesh, k, t_end, num_time_steps, problem)


if __name__ == "__main__":
    main()
