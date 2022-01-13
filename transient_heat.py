import numpy as np

from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import fem
from dolfinx.mesh import create_unit_square, locate_entities_boundary
from dolfinx.io import XDMFFile
from dolfinx.nls import NewtonSolver

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

    v = ufl.TestFunction(V)

    f = fem.Function(V)
    f.interpolate(problem.f)

    rho = fem.Constant(mesh, PETSc.ScalarType(problem.rho()))
    c = fem.Constant(mesh, PETSc.ScalarType(problem.c()))

    F = ufl.inner(rho * c * T_h, v) * ufl.dx + \
        delta_t * ufl.inner(problem.kappa(T_h) *
                            ufl.grad(T_h), ufl.grad(v)) * ufl.dx - \
        ufl.inner(rho * c * T_n + delta_t * f, v) * ufl.dx

    non_lin_problem = fem.NonlinearProblem(F, T_h, [bc])
    solver = NewtonSolver(MPI.COMM_WORLD, non_lin_problem)
    solver.convergence_criterion = "incremental"
    solver.rtol = 1e-6
    solver.report = True

    ksp = solver.krylov_solver
    ksp.setType(PETSc.KSP.Type.PREONLY)
    ksp.getPC().setType(PETSc.PC.Type.LU)

    for n in range(num_time_steps):
        problem.t += delta_t.value

        T_d.interpolate(problem.T)
        f.interpolate(problem.f)

        n, converged = solver.solve(T_h)
        T_h.x.scatter_forward()
        assert(converged)

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
        # Dependence on kappa has been explicitly calculated
        return np.pi * (c * rho * np.cos(np.pi * self.t) + 2 * np.pi *
                        (np.sin(x[0] * np.pi)**2 * np.sin(np.pi * self.t)**2 *
                         np.cos(x[1] * np.pi)**2 + 4.1) *
                        np.sin(np.pi * self.t) - 2 * np.pi *
                        np.sin(x[0] * np.pi)**2 * np.sin(x[1] * np.pi)**2 *
                        np.sin(np.pi * self.t)**3 - 2 * np.pi *
                        np.sin(np.pi * self.t)**3 * np.cos(x[0] * np.pi)**2 *
                        np.cos(x[1] * np.pi)**2) * np.sin(x[0] * np.pi) * \
            np.cos(x[1] * np.pi)

    # Specific heat capacity
    def c(self):
        return 1.3

    # Density
    def rho(self):
        return 2.7

    # Thermal conductivity
    def kappa(self, T):
        # Dummy data representing 4.1 + T**2
        x = np.array([0.0, 0.25, 0.50, 0.75, 1.0])
        y = np.array([4.1, 4.1625, 4.35, 4.6625, 5.1])

        degree = 2
        coeffs = np.polynomial.Polynomial.fit(x, y, degree).convert().coef

        kappa = 0
        for n in range(degree + 1):
            kappa += coeffs[n] * T**n

        return kappa


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
