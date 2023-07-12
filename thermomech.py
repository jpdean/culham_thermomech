# FIXME This needs tidying

import numpy as np

from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import fem, la
from dolfinx.mesh import create_box
from dolfinx.io import XDMFFile
from dolfinx.nls.petsc import NewtonSolver
from dolfinx.common import Timer

import ufl

from utils import TimeDependentExpression, create_mesh_tags_from_locators

from contextlib import ExitStack

import json


def monitor(ksp, its, rnorm):
    print(f"Iteration: {its}, rel. residual: {rnorm}")


def build_nullspace(V):
    """Function to build PETSc nullspace for 2D and 3D elasticity"""

    d = V.mesh.topology.dim

    # Create list of vectors for null space
    index_map = V.dofmap.index_map
    bs = V.dofmap.index_map_bs
    if d == 2:
        num_basis_vecs = 3
    else:
        assert (d == 3)
        num_basis_vecs = 6
    ns = [la.create_petsc_vector(index_map, bs) for i in range(num_basis_vecs)]
    with ExitStack() as stack:
        vec_local = [stack.enter_context(x.localForm()) for x in ns]
        basis = [np.asarray(x) for x in vec_local]

        # Get dof indices for each subspace
        dofs = [V.sub(i).dofmap.list.flatten() for i in range(d)]

        # Build translational nullspace basis
        for i in range(d):
            basis[i][dofs[i]] = 1.0

        # Build rotational nullspace basis
        x = V.tabulate_dof_coordinates()
        dofs_block = V.dofmap.list.flatten()
        x0 = x[dofs_block, 0]
        x1 = x[dofs_block, 1]
        basis[d][dofs[0]] = -x1
        basis[d][dofs[1]] = x0

        if d == 3:
            x2 = x[dofs_block, 2]
            basis[d + 1][dofs[0]] = x2
            basis[d + 1][dofs[2]] = -x0
            basis[d + 2][dofs[2]] = x1
            basis[d + 2][dofs[1]] = -x2

    la.orthonormalize(ns)
    assert la.is_orthonormal(ns)
    return PETSc.NullSpace().create(vectors=ns)


def sigma(v, T, T_ref, alpha_L, E, nu):
    """Compute the stress tensor (elastic and thermal) from displacement"""
    # Elastic strain
    eps_e = ufl.sym(ufl.grad(v))
    # Thermal strain
    eps_T = alpha_L * (T - T_ref) * ufl.Identity(len(v))
    eps = eps_e - eps_T
    # Lame parameters
    mu = E / (2.0 * (1.0 + nu))
    lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return 2.0 * mu * eps + lmbda * ufl.tr(eps) * ufl.Identity(len(v))


def solve(mesh, k, delta_t, num_time_steps, T_0, f_T_expr, f_u, g,
          materials, material_mt, bcs, bc_mt, use_iterative_solver=True,
          write_to_file=False, steps_per_write=10):
    timing_dict = {}
    timer_solve_total = Timer("Solve Total")
    timer_initial_setup = Timer("Initial setup")

    # Simulation time
    t = 0.0
    # Time step
    delta_t = fem.Constant(mesh, PETSc.ScalarType(delta_t))

    # Thermal and elastic function spaces
    V_T = fem.FunctionSpace(mesh, ("Lagrange", k))
    V_u = fem.VectorFunctionSpace(mesh, ("Lagrange", k))

    num_dofs_global = \
        V_T.dofmap.index_map.size_global * V_T.dofmap.index_map_bs + \
        V_u.dofmap.index_map.size_global * V_u.dofmap.index_map_bs

    if write_to_file:
        # FIXME Use one file
        xdmf_file_T = XDMFFile(mesh.comm, "T.xdmf", "w")
        xdmf_file_u = XDMFFile(mesh.comm, "u.xdmf", "w")
        xdmf_file_T.write_mesh(mesh)
        xdmf_file_u.write_mesh(mesh)

    # Create temperature unknown and write initial condition to file
    T_h = fem.Function(V_T)
    T_h.name = "T"
    T_h.interpolate(T_0)
    if write_to_file:
        xdmf_file_T.write_function(T_h, t)
    # Temperature at previous time step
    T_n = fem.Function(V_T)
    T_n.x.array[:] = T_h.x.array

    dx = ufl.Measure("dx", domain=mesh, subdomain_data=material_mt)

    # Thermal problem
    v = ufl.TestFunction(V_T)
    f_T = fem.Function(V_T)
    f_T.interpolate(f_T_expr)
    F_T = - ufl.inner(delta_t * f_T, v) * dx

    # Elastic problem
    u = ufl.TrialFunction(V_u)
    w = ufl.TestFunction(V_u)
    F_u = - ufl.inner(f_u, w) * dx

    # Loop through materials and add terms
    # NOTE This creates a new kernel for every domain marker
    for marker, mat in materials.items():
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
        # Add gravity in the direction of the last component i.e.
        # y dir in 2D, z dir in 3D
        F_u -= ufl.inner(rho * fem.Constant(mesh, g),
                         w[mesh.topology.dim - 1]) * dx(marker)

    # Thermal boundary conditions
    # NOTE Thermal BCs could be time dependent, so keep reference to functions
    bc_funcs_T = []
    for bc in bcs["T"]:
        func = fem.Function(V_T)
        func.interpolate(bc["value"])
        bc_funcs_T.append(func)

    dirichlet_bcs_T = []
    # FIXME Make types enums
    for marker, bc in enumerate(bcs["T"]):
        bc_type = bc["type"]
        ds = ufl.Measure("ds", domain=mesh, subdomain_data=bc_mt["T"])
        if bc_type == "temperature":
            facets = np.array(
                bc_mt["T"].indices[bc_mt["T"].values == marker])
            dofs = fem.locate_dofs_topological(V_T, bc_mt["T"].dim, facets)
            dirichlet_bcs_T.append(
                fem.dirichletbc(bc_funcs_T[marker], dofs))
        elif bc_type == "heat_flux":
            F_T -= delta_t * ufl.inner(bc_funcs_T[marker], v) * ds(marker)
        elif bc_type == "convection":
            T_inf = bc_funcs_T[marker]
            h = bc["h"](T_h)
            F_T += delta_t * ufl.inner(h * (T_h - T_inf), v) * ds(marker)
        else:
            raise Exception(
                f"Boundary condition type {bc_type} not recognised")

    # Elastic boundary conditions
    dirichlet_bcs_u = []
    for marker, bc in enumerate(bcs["u"]):
        bc_type = bc["type"]
        ds = ufl.Measure("ds", domain=mesh, subdomain_data=bc_mt["u"])
        if bc_type == "displacement":
            facets = np.array(
                bc_mt["u"].indices[bc_mt["u"].values == marker])
            dofs = fem.locate_dofs_topological(V_u, bc_mt["u"].dim, facets)
            dirichlet_bcs_u.append(
                fem.dirichletbc(bc["value"], dofs, V_u))
        elif bc_type == "pressure":
            F_u -= ufl.inner(bc["value"] *
                             ufl.FacetNormal(mesh), w) * ds(marker)
        else:
            raise Exception(
                f"Boundary condition type {bc_type} not recognised")

    # Create forms for elastic problem
    a_u = fem.form(ufl.lhs(F_u))
    L_u = fem.form(ufl.rhs(F_u))

    timing_dict["initial_setup"] = mesh.comm.allreduce(
        timer_initial_setup.stop(), op=MPI.MAX)

    # Assemble initial elastic problem
    timer_initial_elastic_assemble = Timer("Initial elastic assemble")
    A_u = fem.petsc.assemble_matrix(a_u, bcs=dirichlet_bcs_u)
    A_u.assemble()
    b_u = fem.petsc.assemble_vector(L_u)
    fem.apply_lifting(b_u, [a_u], bcs=[dirichlet_bcs_u])
    b_u.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    fem.set_bc(b_u, dirichlet_bcs_u)
    timing_dict["initial_elastic_assemble"] = mesh.comm.allreduce(
        timer_initial_elastic_assemble.stop(), op=MPI.MAX)

    # Set up solvers
    timer_solver_setup = Timer("Solver setup")
    non_lin_problem = fem.petsc.NonlinearProblem(F_T, T_h, dirichlet_bcs_T)
    solver = NewtonSolver(mesh.comm, non_lin_problem)
    solver.convergence_criterion = "incremental"
    solver.rtol = 1e-6
    solver.report = True
    ksp_T = solver.krylov_solver
    ksp_u = PETSc.KSP().create(mesh.comm)
    ksp_u.setOperators(A_u)

    if use_iterative_solver:
        # NOTE May need to use GMRES as matrix isn't symmetric due to
        # non-linearity
        opts = PETSc.Options()
        opts[f"{ksp_T.prefix}ksp_type"] = "cg"
        opts[f"{ksp_T.prefix}ksp_rtol"] = 1.0e-8
        opts[f"{ksp_T.prefix}pc_type"] = "hypre"
        opts[f"{ksp_T.prefix}pc_hypre_type"] = "boomeramg"
        opts[f"{ksp_T.prefix}pc_hypre_boomeramg_strong_threshold"] = 0.75
        opts[f"{ksp_T.prefix}pc_hypre_boomeramg_agg_nl"] = 4
        opts[f"{ksp_T.prefix}pc_hypre_boomeramg_agg_num_paths"] = 2
        ksp_T.setFromOptions()
        # ksp_T.view()

        null_space = build_nullspace(V_u)
        A_u.setNearNullSpace(null_space)
        ksp_u.prefix = "ksp_u_"
        opts[f"{ksp_u.prefix}ksp_type"] = "cg"
        opts[f"{ksp_u.prefix}ksp_rtol"] = 1.0e-8
        opts[f"{ksp_u.prefix}pc_type"] = "gamg"
        opts[f"{ksp_u.prefix}pc_gamg_type"] = "agg"
        opts[f"{ksp_u.prefix}pc_gamg_agg_nsmooths"] = 1
        opts[f"{ksp_u.prefix}pc_gamg_threshold"] = 0.015
        opts[f"{ksp_u.prefix}pc_gamg_coarse_eq_limit"] = 1000
        opts[f"{ksp_u.prefix}pc_gamg_square_graph"] = 2
        # TODO Check if I need to update anything
        opts[f"{ksp_u.prefix}pc_gamg_reuse_interpolation"] = 1
        opts[f"{ksp_u.prefix}mg_levels_esteig_ksp_type"] = "cg"
        opts[f"{ksp_u.prefix}mg_levels_ksp_type"] = "chebyshev"
        opts[f"{ksp_u.prefix}mg_levels_pc_type"] = "jacobi"
        opts[f"{ksp_u.prefix}mg_levels_ksp_chebyshev_esteig_steps"] = 20
        opts[f"{ksp_u.prefix}mg_levels_esteig_ksp_max_it"] = 20
        ksp_u.setFromOptions()
        # ksp_u.view()
    else:
        ksp_T.setType(PETSc.KSP.Type.PREONLY)
        ksp_T.getPC().setType(PETSc.PC.Type.LU)

        ksp_u.setType(PETSc.KSP.Type.PREONLY)
        ksp_u.getPC().setType(PETSc.PC.Type.LU)

    timing_dict["solver_setup"] = mesh.comm.allreduce(
        timer_solver_setup.stop(), op=MPI.MAX)

    iters = {"newton": [], "T": [], "u": []}

    # Solve initial elastic problem
    u_h = fem.Function(V_u)
    u_h.name = "u"
    timer_initial_elastic_solve = Timer("Initial elastic solve")
    ksp_u.solve(b_u, u_h.vector)
    timing_dict["initial_elastic_solve"] = mesh.comm.allreduce(
        timer_initial_elastic_solve.stop(), op=MPI.MAX)
    iters["u_init"] = ksp_u.its
    u_h.x.scatter_forward()
    if write_to_file:
        xdmf_file_u.write_function(u_h, t)

    timer_time_steping_loop = Timer("Time stepping loop")
    timing_dict["time_steps"] = {"thermal_solve": [], "elastic_solve": [],
                                 "total": []}
    timer_time_step = Timer("Time step")
    timer_thermal = Timer("Thermal solve")
    timer_elastic = Timer("Elastic solve")
    for n in range(num_time_steps):
        timer_time_step.start()

        t += delta_t.value

        # Update any time dependent functions
        for marker, bc_func in enumerate(bc_funcs_T):
            expr = bcs["T"][marker]["value"]
            if isinstance(expr, TimeDependentExpression):
                expr.t = t
                bc_func.interpolate(expr)
        if isinstance(f_T_expr, TimeDependentExpression):
            f_T_expr.t = t
            f_T.interpolate(f_T_expr)

        # Solve thermal problem
        timer_thermal.start()
        # ksp_T.setMonitor(monitor)
        its, converged = solver.solve(T_h)
        timing_dict["time_steps"]["thermal_solve"].append(mesh.comm.allreduce(
            timer_thermal.stop(), op=MPI.MAX))
        T_h.x.scatter_forward()
        assert (converged)
        iters["newton"].append(its)

        # Solve elastic problem
        A_u.zeroEntries()
        fem.petsc.assemble_matrix(A_u, a_u, bcs=dirichlet_bcs_u)
        A_u.assemble()
        with b_u.localForm() as b_u_loc:
            b_u_loc.set(0)
        fem.petsc.assemble_vector(b_u, L_u)
        fem.apply_lifting(b_u, [a_u], bcs=[dirichlet_bcs_u])
        b_u.ghostUpdate(addv=PETSc.InsertMode.ADD,
                        mode=PETSc.ScatterMode.REVERSE)
        fem.set_bc(b_u, dirichlet_bcs_u)
        # ksp_u.setMonitor(monitor)
        timer_elastic.start()
        ksp_u.solve(b_u, u_h.vector)
        timing_dict["time_steps"]["elastic_solve"].append(mesh.comm.allreduce(
            timer_elastic.stop(), op=MPI.MAX))
        u_h.x.scatter_forward()

        if write_to_file and ((n + 1) % steps_per_write == 0):
            xdmf_file_T.write_function(T_h, t)
            xdmf_file_u.write_function(u_h, t)

        T_n.x.array[:] = T_h.x.array

        iters["T"].append(ksp_T.its)
        iters["u"].append(ksp_u.its)

        timing_dict["time_steps"]["total"].append(mesh.comm.allreduce(
            timer_time_step.stop(), op=MPI.MAX))

    timing_dict["time_stepping_loop"] = mesh.comm.allreduce(
        timer_time_steping_loop.stop(), op=MPI.MAX)

    if write_to_file:
        xdmf_file_T.close()
        xdmf_file_u.close()

    timing_dict["solve_total"] = mesh.comm.allreduce(
        timer_solve_total.stop(), op=MPI.MAX)

    data = {"num_dofs_global": num_dofs_global,
            "iters": iters,
            "timing_dict": timing_dict}
    return {"T": T_h, "u": u_h, "data": data}


def main():
    # TODO Take command line args
    scaling_type = "strong"
    # Approximate number of DOFS (total for strong scaling, per process for
    # weak)
    n_dofs = 20000
    delta_t = 5
    num_time_steps = 5
    # Polynomial order
    k = 1
    # Length of boxmesh
    L = 2.0
    # Width of boxmesh
    w = 1.0

    n_procs = MPI.COMM_WORLD.Get_size()
    if scaling_type == "strong":
        n_total_dofs = n_dofs
    else:
        assert (scaling_type == "weak")
        n_total_dofs = n_procs * n_dofs
    n = round((n_total_dofs / 4)**(1 / 3) - 1)

    mesh = create_box(
        MPI.COMM_WORLD,
        [np.array([0.0, 0.0, 0.0]),
         np.array([L, w, w])],
        [n, n, n])

    # Materials
    from materials import materials as mat_dict
    materials = {0: mat_dict["304SS"],
                 1: mat_dict["Copper"],
                 2: mat_dict["CuCrZr"]}
    # Create material meshtags, making them align with mesh
    x_1 = round(n / 4) * L / n
    x_2 = round(n / 2) * L / n
    material_mt = create_mesh_tags_from_locators(
        mesh,
        {0: lambda x: x[0] <= x_1,
         1: lambda x: np.logical_and(x[0] >= x_1, x[0] <= x_2),
         2: lambda x: x[0] >= x_2},
        mesh.topology.dim)

    # Specify boundary conditions
    bcs = {}
    bcs["T"] = [{"type": "temperature",
                 "value": lambda x: 293.15 * np.ones_like(x[0])},
                {"type": "convection",
                 "value": lambda x: 293.15 * np.ones_like(x[0]),
                 "h": lambda T: 5},
                {"type": "convection",
                 "value": lambda x: 293.15 * np.ones_like(x[0]),
                 "h": mat_dict["water"]["h"]},
                {"type": "heat_flux",
                 "value": lambda x: 1e5 * np.ones_like(x[0])}]
    bcs["u"] = [{"type": "displacement",
                 "value": np.array([0, 0, 0], dtype=PETSc.ScalarType)},
                {"type": "pressure",
                 "value": fem.Constant(mesh, PETSc.ScalarType(-1e6))}]
    # Create meshtags for boundary conditions
    bc_mt = {}
    bc_mt["T"] = create_mesh_tags_from_locators(
        mesh,
        {0: lambda x: np.isclose(x[0], 0.0),
         1: lambda x: np.logical_or(np.isclose(x[1], 0.0),
                                 np.isclose(x[2], 0.0)),
         2: lambda x: np.logical_or(np.isclose(x[1], w),
                                 np.isclose(x[2], w)),
         3: lambda x: np.isclose(x[0], L)},
        mesh.topology.dim - 1)
    bc_mt["u"] = create_mesh_tags_from_locators(
        mesh,
        {0: lambda x: np.isclose(x[0], 0.0),
         1: lambda x: np.isclose(x[1], w)},
        mesh.topology.dim - 1)

    # Elastic source function (not including gravity)
    f_u = fem.Constant(mesh, np.array([0, 0, 0], dtype=PETSc.ScalarType))

    # Thermal source function
    def f_T(x): return np.zeros_like(x[0])

    # Initial temperature
    def T_0(x): return 293.15 * np.ones_like(x[0])

    # Acceleration due to gravity
    g = PETSc.ScalarType(- 9.81)

    # Solve the problem
    results = solve(mesh, k, delta_t, num_time_steps, T_0, f_T,
                    f_u, g, materials, material_mt, bcs, bc_mt,
                    write_to_file=True, steps_per_write=1)

    # Save timing and iteration count data to JSON
    if mesh.comm.Get_rank() == 0:
        with open(f"thermomech_{n_procs}.json", "w") as f:
            json.dump(results["data"], f)


if __name__ == "__main__":
    main()
