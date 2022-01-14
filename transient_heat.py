import numpy as np

from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import fem
from dolfinx.mesh import (create_unit_square, locate_entities_boundary,
                          locate_entities, MeshTags)
from dolfinx.io import XDMFFile
from dolfinx.nls import NewtonSolver

import ufl


def ufl_poly_from_table_data(x, y, degree, u):
    """Given a list of point data x and y, this function returns a fitted
    polynomial of degree `degree` in terms of the UFL `Function` `u`"""
    coeffs = np.polynomial.Polynomial.fit(x, y, degree).convert().coef

    poly = 0
    for n in range(degree + 1):
        poly += coeffs[n] * u**n
    return poly


def solve(mesh, k, t_end, num_time_steps, problem):
    V = fem.FunctionSpace(mesh, ("Lagrange", k))

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

    rho = problem.rho(T_h)
    c = problem.c(T_h)
    kappa = problem.kappa(T_h)
    F = ufl.inner(rho * c * T_h, v) * ufl.dx + \
        delta_t * ufl.inner(kappa * ufl.grad(T_h), ufl.grad(v)) * ufl.dx - \
        ufl.inner(rho * c * T_n + delta_t * f, v) * ufl.dx

    bcs, boundary_mt = problem.bcs(mesh)
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=boundary_mt)

    bc_funcs = []
    for bc in bcs:
        func = fem.Function(V)
        func.interpolate(bc["value"])
        bc_funcs.append(func)

    dirichlet_bcs = []
    # FIXME Make types enums
    for marker, bc in enumerate(bcs):
        bc_type = bc["type"]
        if bc_type == "dirichlet":
            facets = np.array(
                boundary_mt.indices[boundary_mt.values == marker])
            dofs = fem.locate_dofs_topological(V, boundary_mt.dim, facets)
            dirichlet_bcs.append(
                fem.dirichletbc(bc_funcs[marker], dofs))
        elif bc_type == "neumann":
            F -= delta_t * ufl.inner(bc_funcs[marker], v) * ds(marker)
        else:
            raise Exception(
                f"Boundary condition type {bc_type} not recognised")

    non_lin_problem = fem.NonlinearProblem(F, T_h, dirichlet_bcs)
    solver = NewtonSolver(MPI.COMM_WORLD, non_lin_problem)
    solver.convergence_criterion = "incremental"
    solver.rtol = 1e-6
    solver.report = True

    ksp = solver.krylov_solver
    ksp.setType(PETSc.KSP.Type.PREONLY)
    ksp.getPC().setType(PETSc.PC.Type.LU)

    for n in range(num_time_steps):
        problem.t += delta_t.value

        for marker, bc_func in enumerate(bc_funcs):
            bc_func.interpolate(bcs[marker]["value"])
        f.interpolate(problem.f)

        its, converged = solver.solve(T_h)
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
        # Dependence on material parameters has been explicitly computed
        return np.pi * ((np.sin(x[0] * np.pi)**2 * np.sin(np.pi * self.t)**2 *
                         np.cos(x[1] * np.pi)**2 + 1.3) *
                        (np.sin(x[0] * np.pi)**2 * np.sin(np.pi * self.t)**2
                         * np.cos(x[1] * np.pi)**2 + 2.7) *
                        np.cos(np.pi * self.t) + 2 * np.pi *
                        (np.sin(x[0] * np.pi)**2 * np.sin(np.pi * self.t)**2 *
                        np.cos(x[1] * np.pi)**2 + 4.1) * np.sin(np.pi * self.t)
                        - 2 * np.pi * np.sin(x[0] * np.pi)**2 * np.sin(x[1] *
                        np.pi)**2 * np.sin(np.pi * self.t)**3 - 2 * np.pi *
                        np.sin(np.pi * self.t)**3 * np.cos(x[0] * np.pi)**2 *
                        np.cos(x[1] * np.pi)**2) * np.sin(x[0] * np.pi) * \
            np.cos(x[1] * np.pi)

    # Specific heat capacity
    def c(self, T):
        # Dummy data representing 1.3 + T**2
        x = np.array([0.0, 0.25, 0.50, 0.75, 1.0])
        y = np.array([1.3, 1.3625, 1.55, 1.8625, 2.3])
        return ufl_poly_from_table_data(x, y, 2, T)

    # Density
    def rho(self, T):
        # Dummy data representing 2.7 + T**2
        x = np.array([0.0, 0.25, 0.50, 0.75, 1.0])
        y = np.array([2.7, 2.7625, 2.95, 3.2625, 3.7])
        return ufl_poly_from_table_data(x, y, 2, T)

    # Thermal conductivity
    def kappa(self, T):
        # Dummy data representing 4.1 + T**2
        x = np.array([0.0, 0.25, 0.50, 0.75, 1.0])
        y = np.array([4.1, 4.1625, 4.35, 4.6625, 5.1])
        return ufl_poly_from_table_data(x, y, 2, T)

    def bcs(self, mesh):
        boundaries = [lambda x: np.isclose(x[0], 0),
                      lambda x: np.isclose(x[0], 1),
                      lambda x: np.isclose(x[1], 0),
                      lambda x: np.isclose(x[1], 1)]

        def neumann_bc(self, x):
            # NOTE This is just the Neumann BC for the right boundary
            # TODO Implement with UFL instead?
            return np.pi * (np.sin(x[0] * np.pi)**2 *
                            np.sin(np.pi * self.t)**2 *
                            np.cos(x[1] * np.pi)**2 + 4.1) * \
                np.sin(np.pi * self.t) * np.cos(x[0] * np.pi) \
                * np.cos(x[1] * np.pi)

        bcs = [{"type": "dirichlet",
                "value": self.T},
               {"type": "neumann",
                "value": neumann_bc},
               {"type": "dirichlet",
                "value": self.T},
               {"type": "dirichlet",
                "value": self.T}]

        facet_indices, facet_markers = [], []
        fdim = mesh.topology.dim - 1
        # Use index in the `boundaries` list as the unique marker
        for marker, locator in enumerate(boundaries):
            # FIXME Use locate entities boundary?
            facets = locate_entities(mesh, fdim, locator)
            facet_indices.append(facets)
            facet_markers.append(np.full(len(facets), marker))
        facet_indices = np.array(np.hstack(facet_indices), dtype=np.int32)
        facet_markers = np.array(np.hstack(facet_markers), dtype=np.int32)
        sorted_facets = np.argsort(facet_indices)
        mt = MeshTags(mesh, fdim, facet_indices[sorted_facets],
                      facet_markers[sorted_facets])
        return bcs, mt


def main():
    t_end = 2
    num_time_steps = 100
    n = 32
    k = 1
    # TODO Use rectangle mesh
    mesh = create_unit_square(MPI.COMM_WORLD, n, n)
    problem = Problem()

    solve(mesh, k, t_end, num_time_steps, problem)


if __name__ == "__main__":
    main()
