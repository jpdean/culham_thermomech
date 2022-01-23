# FIXME This needs tidying

import numpy as np

from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import fem
from dolfinx.mesh import create_box
from dolfinx.io import XDMFFile
from dolfinx.nls import NewtonSolver

import ufl

from problems import (TimeDependentExpression, create_mesh_tags,
                      ufl_poly_from_table_data)


def sigma(v, T, T_ref, alpha_L, E, nu):
    # Elastic strain
    eps_e = ufl.sym(ufl.grad(v))
    eps_T = alpha_L * (T - T_ref) * ufl.Identity(len(v))
    eps = eps_e - eps_T
    mu = E / (2.0 * (1.0 + nu))
    lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return 2.0 * mu * eps + lmbda * ufl.tr(eps) * ufl.Identity(len(v))


def solve(mesh, k, t_end, num_time_steps, T_i, f_T_expr, f_u, materials,
          material_mt, bcs, bc_mt, use_iterative_solver=True):
    t = 0.0
    V_T = fem.FunctionSpace(mesh, ("Lagrange", k))
    V_u = fem.VectorFunctionSpace(mesh, ("Lagrange", k))

    # Time step
    delta_t = fem.Constant(mesh, PETSc.ScalarType(t_end / num_time_steps))

    # FIXME Use one file
    xdmf_file_T = XDMFFile(MPI.COMM_WORLD, "T.xdmf", "w")
    xdmf_file_u = XDMFFile(MPI.COMM_WORLD, "u.xdmf", "w")
    xdmf_file_T.write_mesh(mesh)
    xdmf_file_u.write_mesh(mesh)

    T_h = fem.Function(V_T)
    T_h.name = "T"
    T_h.interpolate(T_i)
    xdmf_file_T.write_function(T_h, t)

    T_n = fem.Function(V_T)
    T_n.x.array[:] = T_h.x.array

    v = ufl.TestFunction(V_T)

    u = ufl.TrialFunction(V_u)
    w = ufl.TestFunction(V_u)

    f_T = fem.Function(V_T)
    f_T.interpolate(f_T_expr)

    F_T = - ufl.inner(delta_t * f_T, v) * ufl.dx
    F_u = - ufl.inner(f_u, w) * ufl.dx

    dx = ufl.Measure("dx", domain=mesh, subdomain_data=material_mt)

    # FIXME I think this creates a new kernel for every material
    for marker, mat in enumerate(materials):
        c = mat["c"](T_h)
        rho = mat["rho"](T_h)
        kappa = mat["kappa"](T_h)
        F_T += ufl.inner(rho * c * T_h, v) * dx(marker) + \
            delta_t * ufl.inner(kappa * ufl.grad(T_h),
                                ufl.grad(v)) * dx(marker) - \
            ufl.inner(rho * c * T_n, v) * dx(marker)

        (alpha_L, T_ref) = mat["thermal_strain"]
        E = mat["E"]
        nu = mat["nu"]
        F_u += ufl.inner(
            sigma(u, T_h, T_ref, alpha_L(T_h), E(T_h), nu),
            ufl.grad(w)) * dx(marker)

    ds = ufl.Measure("ds", domain=mesh, subdomain_data=bc_mt)

    # Thermal BCs could be time dependent, so keep reference to functions
    bc_funcs_T = []
    for bc in bcs:
        if bc["type"] in ["temperature", "heat_flux", "convection"]:
            func = fem.Function(V_T)
            func.interpolate(bc["value"])
            bc_funcs_T.append(func)

    dirichlet_bcs_T = []
    dirichlet_bcs_u = []
    # FIXME Make types enums
    for marker, bc in enumerate(bcs):
        bc_type = bc["type"]
        if bc_type == "temperature":
            facets = np.array(
                bc_mt.indices[bc_mt.values == marker])
            dofs = fem.locate_dofs_topological(V_T, bc_mt.dim, facets)
            dirichlet_bcs_T.append(
                fem.dirichletbc(bc_funcs_T[marker], dofs))
        elif bc_type == "heat_flux":
            F_T -= delta_t * ufl.inner(bc_funcs_T[marker], v) * ds(marker)
        elif bc_type == "convection":
            T_inf = bc_funcs_T[marker]
            h = bc["h"](T_h)
            F_T += delta_t * ufl.inner(h * (T_h - T_inf), v) * ds(marker)
        elif bc_type == "displacement":
            facets = np.array(
                bc_mt.indices[bc_mt.values == marker])
            dofs = fem.locate_dofs_topological(V_u, bc_mt.dim, facets)
            dirichlet_bcs_u.append(
                fem.dirichletbc(bc["value"], dofs, V_u))
        elif bc_type == "pressure":
            F_u -= ufl.inner(bc["value"] *
                             ufl.FacetNormal(mesh), w) * ds(marker)
        else:
            raise Exception(
                f"Boundary condition type {bc_type} not recognised")

    a_u = fem.form(ufl.lhs(F_u))
    L_u = fem.form(ufl.rhs(F_u))

    A_u = fem.assemble_matrix(a_u, bcs=dirichlet_bcs_u)
    A_u.assemble()

    b_u = fem.assemble_vector(L_u)
    fem.apply_lifting(b_u, [a_u], bcs=[dirichlet_bcs_u])
    b_u.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    fem.set_bc(b_u, dirichlet_bcs_u)

    ksp_u = PETSc.KSP().create(MPI.COMM_WORLD)
    ksp_u.setOperators(A_u)
    # TODO Use iterative
    ksp_u.setType(PETSc.KSP.Type.PREONLY)
    ksp_u.getPC().setType(PETSc.PC.Type.LU)

    u_h = fem.Function(V_u)
    u_h.name = "u"
    ksp_u.solve(b_u, u_h.vector)
    u_h.x.scatter_forward()
    xdmf_file_u.write_function(u_h, t)

    non_lin_problem = fem.NonlinearProblem(F_T, T_h, dirichlet_bcs_T)
    solver = NewtonSolver(MPI.COMM_WORLD, non_lin_problem)
    solver.convergence_criterion = "incremental"
    solver.rtol = 1e-6
    solver.report = True

    ksp_T = solver.krylov_solver

    if use_iterative_solver:
        # NOTE May need to use GMRES as matrix isn't symmetric due to
        # non-linearity
        ksp_T.setType(PETSc.KSP.Type.CG)
        ksp_T.setTolerances(rtol=1.0e-12)
        ksp_T.getPC().setType(PETSc.PC.Type.HYPRE)
        ksp_T.getPC().setHYPREType("boomeramg")
    else:
        ksp_T.setType(PETSc.KSP.Type.PREONLY)
        ksp_T.getPC().setType(PETSc.PC.Type.LU)
    # viewer = PETSc.Viewer().createASCII("viewer.txt")
    # ksp_T.view(viewer)

    for n in range(num_time_steps):
        t += delta_t.value

        for marker, bc_func in enumerate(bc_funcs_T):
            expr = bcs[marker]["value"]
            if isinstance(expr, TimeDependentExpression):
                expr.t = t
                bc_func.interpolate(expr)
        if isinstance(f_T_expr, TimeDependentExpression):
            f_T_expr.t = t
            f_T.interpolate(f_T_expr)

        its, converged = solver.solve(T_h)
        T_h.x.scatter_forward()
        assert(converged)

        A_u.zeroEntries()
        fem.assemble_matrix(A_u, a_u, bcs=dirichlet_bcs_u)
        A_u.assemble()
        with b_u.localForm() as b_u_loc:
            b_u_loc.set(0)
        fem.assemble_vector(b_u, L_u)
        fem.apply_lifting(b_u, [a_u], bcs=[dirichlet_bcs_u])
        b_u.ghostUpdate(addv=PETSc.InsertMode.ADD,
                        mode=PETSc.ScatterMode.REVERSE)
        fem.set_bc(b_u, dirichlet_bcs_u)

        ksp_u.solve(b_u, u_h.vector)
        u_h.x.scatter_forward()

        xdmf_file_T.write_function(T_h, t)
        xdmf_file_u.write_function(u_h, t)

        T_n.x.array[:] = T_h.x.array

    xdmf_file_T.close()
    xdmf_file_u.close()

    return (T_h, u_h)


def main():
    t_end = 1
    num_time_steps = 20
    n = 16
    k = 1
    mesh = create_box(
        MPI.COMM_WORLD, [np.array([0.0, 0.0, 0.0]),
                         np.array([2.0, 1.0, 1.0])], [n, n, n])

    def T_i(x):
        return np.zeros_like(x[0])

    f_T_expr = TimeDependentExpression(
        lambda x, t: np.sin(np.pi * x[0]) * np.cos(np.pi * x[1]) *
        np.sin(np.pi * t))

    materials = []
    # TODO Test ufl_poly_from_table_data for elastic properties
    materials.append({"name": "mat_1",
                      "c": lambda T: 1.3 + T**2,
                      "rho": lambda T: 2.7 + T**2,
                      "kappa": lambda T: 4.1 + T**2,
                      "nu": 0.33,
                      "E": lambda T: 1.0 + 0.1 * T**2,
                      "thermal_strain": (lambda T: 0.1 + 0.01 * T**3, 1.5)})
    materials.append({"name": "mat_2",
                      "c": lambda T: 1.7 + T**2,
                      "rho": lambda T: 0.7 + 0.1 * T**2,
                      "kappa": lambda T: 3.2 + 0.6 * T**2,
                      "nu": 0.1,
                      "E": lambda T: 1.0 + 0.5 * T**2,
                      "thermal_strain": (lambda T: 0.2 + 0.015 * T**2, 1.0)})
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

    bcs = [{"type": "convection",
            "value": lambda x: 0.1 * np.ones_like(x[0]),
            "h": h},
           {"type": "heat_flux",
            "value": lambda x: 0.5 * np.ones_like(x[0])},
           {"type": "temperature",
            "value": lambda x: np.zeros_like(x[0])},
           {"type": "displacement",
            "value": np.array([0, 0], dtype=PETSc.ScalarType)},
           {"type": "pressure",
            "value": fem.Constant(mesh, PETSc.ScalarType(-1))}]

    bc_mt = create_mesh_tags(
        mesh,
        [lambda x: np.logical_or(np.isclose(x[1], 0), np.isclose(x[1], 1)),
         lambda x: np.isclose(x[2], 1),
         lambda x: np.isclose(x[0], 0),
         lambda x: np.isclose(x[0], 0),
         lambda x: np.isclose(x[2], 1.0)],
        mesh.topology.dim - 1)

    f_u = fem.Constant(mesh, np.array([0, 0, -1], dtype=PETSc.ScalarType))

    solve(mesh, k, t_end, num_time_steps, T_i, f_T_expr, f_u, materials,
          material_mt, bcs, bc_mt)


if __name__ == "__main__":
    main()
