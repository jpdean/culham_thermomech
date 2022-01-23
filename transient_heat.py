from ast import Constant
import numpy as np

from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import fem
from dolfinx.mesh import create_unit_square
from dolfinx.io import XDMFFile
from dolfinx.nls import NewtonSolver

import ufl

from problems import (TimeDependentExpression, create_mesh_tags,
                      ufl_poly_from_table_data)


def to_theta(v, v_n, theta):
    return (1 - theta) * v_n + theta * v


def solve(mesh, k, t_end, num_time_steps, T_i, f_expr, materials, material_mt,
          bcs, bc_mt, use_iterative_solver=True):
    t = 0.0
    V = fem.FunctionSpace(mesh, ("Lagrange", k))

    # Time step
    delta_t = fem.Constant(mesh, PETSc.ScalarType(t_end / num_time_steps))

    xdmf_file = XDMFFile(MPI.COMM_WORLD, "T.xdmf", "w")
    xdmf_file.write_mesh(mesh)

    # FIXME Could just interpolate T_n and f_n
    T_h = fem.Function(V)
    T_h.name = "T"
    T_h.interpolate(T_i)
    xdmf_file.write_function(T_h, t)

    T_n = fem.Function(V)
    T_n.x.array[:] = T_h.x.array

    # theta = Constant(mesh, PETSc.ScalarType(0.5))
    theta = 0.5
    T_theta = to_theta(T_h, T_n, theta)

    v = ufl.TestFunction(V)

    f = fem.Function(V)
    f.interpolate(f_expr)
    f_n = fem.Function(V)
    f_n.x.array[:] = f.x.array
    f_theta = to_theta(f, f_n, theta)

    F = - ufl.inner(delta_t * f_theta, v) * ufl.dx

    dx = ufl.Measure("dx", domain=mesh, subdomain_data=material_mt)

    # FIXME I think this creates a new kernel for every material
    for marker, mat in enumerate(materials):
        c = mat["c"](T_theta)
        rho = mat["rho"](T_theta)
        kappa = mat["kappa"](T_theta)
        F += ufl.inner(rho * c * T_h, v) * dx(marker) + \
            delta_t * ufl.inner(kappa * ufl.grad(T_theta),
                                ufl.grad(v)) * dx(marker) - \
            ufl.inner(rho * c * T_n, v) * dx(marker)

    ds = ufl.Measure("ds", domain=mesh, subdomain_data=bc_mt)

    bc_funcs = []
    for bc in bcs:
        func = fem.Function(V)
        func.interpolate(bc["value"])
        func_n = fem.Function(V)
        func_n.x.array[:] = func.x.array
        bc_funcs.append((func, func_n))

    dirichlet_bcs = []
    # FIXME Make types enums
    for marker, bc in enumerate(bcs):
        bc_type = bc["type"]
        if bc_type == "dirichlet":
            facets = np.array(
                bc_mt.indices[bc_mt.values == marker])
            dofs = fem.locate_dofs_topological(V, bc_mt.dim, facets)
            # bc_func_theta = to_theta(bc_funcs[marker][0],
            #                          bc_funcs[marker][1],
            #                          theta)
            dirichlet_bcs.append(
                fem.dirichletbc(bc_funcs[marker][0], dofs))
        elif bc_type == "neumann":
            bc_func_theta = to_theta(bc_funcs[marker][0],
                                     bc_funcs[marker][1],
                                     theta)
            F -= delta_t * ufl.inner(bc_func_theta, v) * ds(marker)
        elif bc_type == "robin":
            T_inf_theta = to_theta(bc_funcs[marker][0],
                                   bc_funcs[marker][1],
                                   theta)
            h = bc["h"](T_theta)
            F += delta_t * ufl.inner(h * (T_theta - T_inf_theta), v) * ds(marker)
        else:
            raise Exception(
                f"Boundary condition type {bc_type} not recognised")

    non_lin_problem = fem.NonlinearProblem(F, T_h, dirichlet_bcs)
    solver = NewtonSolver(MPI.COMM_WORLD, non_lin_problem)
    solver.convergence_criterion = "incremental"
    solver.rtol = 1e-6
    solver.report = True

    ksp = solver.krylov_solver

    if use_iterative_solver:
        # NOTE May need to use GMRES as matrix isn't symmetric due to
        # non-linearity
        ksp.setType(PETSc.KSP.Type.CG)
        ksp.setTolerances(rtol=1.0e-12)
        ksp.getPC().setType(PETSc.PC.Type.HYPRE)
        ksp.getPC().setHYPREType("boomeramg")
    else:
        ksp.setType(PETSc.KSP.Type.PREONLY)
        ksp.getPC().setType(PETSc.PC.Type.LU)
    viewer = PETSc.Viewer().createASCII("viewer.txt")
    ksp.view(viewer)

    for n in range(num_time_steps):
        t += delta_t.value

        for marker, bc_func in enumerate(bc_funcs):
            expr = bcs[marker]["value"]
            if isinstance(expr, TimeDependentExpression):
                expr.t = t
                bc_func[0].interpolate(expr)
        if isinstance(f_expr, TimeDependentExpression):
            f_expr.t = t
            f.interpolate(f_expr)

        its, converged = solver.solve(T_h)
        T_h.x.scatter_forward()
        assert(converged)

        T_n.x.array[:] = T_h.x.array
        for marker, bc_func in enumerate(bc_funcs):
            bc_func[1].x.array[:] = bc_func[0].x.array

        xdmf_file.write_function(T_h, t)

    xdmf_file.close()

    return T_h


def main():
    t_end = 2
    num_time_steps = 100
    n = 32
    k = 1
    # TODO Use rectangle mesh
    mesh = create_unit_square(MPI.COMM_WORLD, n, n)

    def T_i(x):
        return np.zeros_like(x[0])

    f = TimeDependentExpression(
        lambda x, t: np.sin(np.pi * x[0]) * np.cos(np.pi * x[1]) *
        np.sin(np.pi * t))

    materials = []
    # TODO Test ufl_poly_from_table_data for elastic properties
    materials.append({"name": "mat_1",
                      "c": lambda T: 1.3 + T**2,
                      "rho": lambda T: 2.7 + T**2,
                      "kappa": lambda T: 4.1 + T**2})
    materials.append({"name": "mat_2",
                      "c": lambda T: 1.7 + T**2,
                      "rho": lambda T: 0.7 + 0.1 * T**2,
                      "kappa": lambda T: 3.2 + 0.6 * T**2})
    material_mt = create_mesh_tags(
        mesh,
        [lambda x: x[0] <= 0.5,
         lambda x: x[0] >= 0.5],
        mesh.topology.dim)

    def h(T):
        # Test ufl.conditional works OK for complicated coefficients
        # which should be approximated with multiple polynomials.
        # Dummy data representing 2.7 + T**2
        x = np.array([0.0, 0.25, 0.50, 0.75, 1.0])
        y = np.array([3.5, 3.5625, 3.75, 4.0625, 4.5])
        h_poly = ufl_poly_from_table_data(x, y, 2, T)
        # NOTE For this problem, this will always be false as the solution
        # is zero on this boundary
        return ufl.conditional(T > 0.5, 3.5 + T**2, h_poly)

    bcs = [{"type": "robin",
            "value": lambda x: 0.1 * np.ones_like(x[0]),
            "h": h},
           {"type": "neumann",
            "value": lambda x: 0.5 * np.ones_like(x[0])},
           {"type": "dirichlet",
            "value": lambda x: np.zeros_like(x[0])},
           {"type": "dirichlet",
            "value": lambda x: np.zeros_like(x[0])}]

    bc_mt = create_mesh_tags(
        mesh,
        [lambda x: np.isclose(x[0], 0),
         lambda x: np.isclose(x[0], 1),
         lambda x: np.isclose(x[1], 0),
         lambda x: np.isclose(x[1], 1)],
        mesh.topology.dim - 1)

    solve(mesh, k, t_end, num_time_steps, T_i, f, materials, material_mt,
          bcs, bc_mt)


if __name__ == "__main__":
    main()
