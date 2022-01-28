# FIXME This needs tidying

import numpy as np

from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import fem, la
from dolfinx.mesh import create_box
from dolfinx.io import XDMFFile
from dolfinx.nls import NewtonSolver

import ufl

from utils import TimeDependentExpression, create_mesh_tags_from_locators

from contextlib import ExitStack


def monitor(ksp, its, rnorm):
    print(f"Iteration: {its}, rel. residual: {rnorm}")


def build_nullspace(V):
    # TODO This can be simplified
    """Function to build PETSc nullspace for 2D and 3D elasticity"""

    d = V.mesh.topology.dim

    # Create list of vectors for null space
    index_map = V.dofmap.index_map
    bs = V.dofmap.index_map_bs
    if d == 2:
        num_basis_vecs = 3
    else:
        assert(d == 3)
        num_basis_vecs = 6
    ns = [la.create_petsc_vector(index_map, bs) for i in range(num_basis_vecs)]
    with ExitStack() as stack:
        vec_local = [stack.enter_context(x.localForm()) for x in ns]
        basis = [np.asarray(x) for x in vec_local]

        # Get dof indices for each subspace
        dofs = [V.sub(i).dofmap.list.array for i in range(d)]

        # Build translational nullspace basis
        for i in range(d):
            basis[i][dofs[i]] = 1.0

        # Build rotational nullspace basis
        x = V.tabulate_dof_coordinates()
        dofs_block = V.dofmap.list.array
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
    # Elastic strain
    eps_e = ufl.sym(ufl.grad(v))
    eps_T = alpha_L * (T - T_ref) * ufl.Identity(len(v))
    eps = eps_e - eps_T
    mu = E / (2.0 * (1.0 + nu))
    lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return 2.0 * mu * eps + lmbda * ufl.tr(eps) * ufl.Identity(len(v))


def solve(mesh, k, t_end, num_time_steps, T_i, f_T_expr, f_u, g,
          materials, material_mt, bcs, bc_mt, use_iterative_solver=True,
          write_to_file=True):
    t = 0.0
    V_T = fem.FunctionSpace(mesh, ("Lagrange", k))
    V_u = fem.VectorFunctionSpace(mesh, ("Lagrange", k))

    if mesh.comm.Get_rank() == 0:
        num_dofs_global = \
            V_T.dofmap.index_map.size_global * V_T.dofmap.index_map_bs + \
            V_u.dofmap.index_map.size_global * V_u.dofmap.index_map_bs
        print(f"Number of DOFs (global): {num_dofs_global}")

    # Time step
    delta_t = fem.Constant(mesh, PETSc.ScalarType(t_end / num_time_steps))

    if write_to_file:
        # FIXME Use one file
        xdmf_file_T = XDMFFile(MPI.COMM_WORLD, "T.xdmf", "w")
        xdmf_file_u = XDMFFile(MPI.COMM_WORLD, "u.xdmf", "w")
        xdmf_file_T.write_mesh(mesh)
        xdmf_file_u.write_mesh(mesh)

    T_h = fem.Function(V_T)
    T_h.name = "T"
    T_h.interpolate(T_i)
    if write_to_file:
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
        # Add gravity in the direction of the last component i.e.
        # y dir in 2D, z dir in 3D
        F_u -= ufl.inner(rho * fem.Constant(mesh, g),
                         w[mesh.topology.dim - 1]) * dx(marker)

    # Thermal BCs could be time dependent, so keep reference to functions
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

    a_u = fem.form(ufl.lhs(F_u))
    L_u = fem.form(ufl.rhs(F_u))

    A_u = fem.assemble_matrix(a_u, bcs=dirichlet_bcs_u)
    A_u.assemble()

    b_u = fem.assemble_vector(L_u)
    fem.apply_lifting(b_u, [a_u], bcs=[dirichlet_bcs_u])
    b_u.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    fem.set_bc(b_u, dirichlet_bcs_u)

    non_lin_problem = fem.NonlinearProblem(F_T, T_h, dirichlet_bcs_T)
    solver = NewtonSolver(MPI.COMM_WORLD, non_lin_problem)
    solver.convergence_criterion = "incremental"
    solver.rtol = 1e-6
    solver.report = True

    ksp_T = solver.krylov_solver
    ksp_u = PETSc.KSP().create(MPI.COMM_WORLD)
    ksp_u.setOperators(A_u)

    if use_iterative_solver:
        # NOTE May need to use GMRES as matrix isn't symmetric due to
        # non-linearity
        ksp_T.setType(PETSc.KSP.Type.CG)
        ksp_T.setTolerances(rtol=1.0e-12)
        ksp_T.getPC().setType(PETSc.PC.Type.HYPRE)
        ksp_T.getPC().setHYPREType("boomeramg")

        null_space = build_nullspace(V_u)
        A_u.setNearNullSpace(null_space)
        opts = PETSc.Options()
        opts["ksp_type"] = "cg"
        opts["ksp_rtol"] = 1.0e-12
        opts["pc_type"] = "gamg"
        opts["mg_levels_ksp_type"] = "chebyshev"
        opts["mg_levels_pc_type"] = "jacobi"
        opts["mg_levels_esteig_ksp_type"] = "cg"
        opts["mg_levels_ksp_chebyshev_esteig_steps"] = 20
        ksp_u.setFromOptions()
    else:
        ksp_T.setType(PETSc.KSP.Type.PREONLY)
        ksp_T.getPC().setType(PETSc.PC.Type.LU)

        ksp_u.setType(PETSc.KSP.Type.PREONLY)
        ksp_u.getPC().setType(PETSc.PC.Type.LU)
    # viewer = PETSc.Viewer().createASCII("viewer.txt")
    # ksp_T.view(viewer)
    # ksp_u.view(viewer)

    u_h = fem.Function(V_u)
    u_h.name = "u"
    ksp_u.solve(b_u, u_h.vector)
    u_h.x.scatter_forward()
    if write_to_file:
        xdmf_file_u.write_function(u_h, t)

    for n in range(num_time_steps):
        t += delta_t.value

        for marker, bc_func in enumerate(bc_funcs_T):
            expr = bcs["T"][marker]["value"]
            if isinstance(expr, TimeDependentExpression):
                expr.t = t
                bc_func.interpolate(expr)
        if isinstance(f_T_expr, TimeDependentExpression):
            f_T_expr.t = t
            f_T.interpolate(f_T_expr)

        # ksp_T.setMonitor(monitor)
        its, converged = solver.solve(T_h)
        # if mesh.comm.Get_rank() == 0:
        #     print(its)
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

        # ksp_u.setMonitor(monitor)
        ksp_u.solve(b_u, u_h.vector)
        u_h.x.scatter_forward()

        if write_to_file:
            xdmf_file_T.write_function(T_h, t)
            xdmf_file_u.write_function(u_h, t)

        T_n.x.array[:] = T_h.x.array

    if write_to_file:
        xdmf_file_T.close()
        xdmf_file_u.close()

    return (T_h, u_h)


def main():
    t_end = 750
    num_time_steps = 10
    n = 16
    k = 1
    L = 2.0
    w = 1.0

    # FIXME Mesh does not necessarily align with materials
    mesh = create_box(
        MPI.COMM_WORLD,
        [np.array([0.0, 0.0, 0.0]),
         np.array([L, w, w])],
        [n, n, n])

    # TODO Let solver take dictionary of materials instead of list
    from materials import materials as mat_dict
    materials = []
    materials.append(mat_dict["304SS"])
    materials.append(mat_dict["Copper"])
    materials.append(mat_dict["CuCrZr"])

    material_mt = create_mesh_tags_from_locators(
        mesh,
        [lambda x: x[0] <= L / 4,
         lambda x: np.logical_and(x[0] >= L / 4, x[0] <= L / 2),
         lambda x: x[0] >= L / 2],
        mesh.topology.dim)

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

    bc_mt = {}
    bc_mt["T"] = create_mesh_tags_from_locators(
        mesh,
        [lambda x: np.isclose(x[0], 0.0),
         lambda x: np.logical_or(np.isclose(x[1], 0.0),
                                 np.isclose(x[2], 0.0)),
         lambda x: np.logical_or(np.isclose(x[1], w),
                                 np.isclose(x[2], w)),
         lambda x: np.isclose(x[0], L)],
        mesh.topology.dim - 1)
    bc_mt["u"] = create_mesh_tags_from_locators(
        mesh,
        [lambda x: np.isclose(x[0], 0.0),
         lambda x: np.isclose(x[1], w)],
        mesh.topology.dim - 1)

    f_u = fem.Constant(mesh, np.array([0, 0, 0], dtype=PETSc.ScalarType))

    def f_T(x): return np.zeros_like(x[0])

    def T_i(x): return 293.15 * np.ones_like(x[0])

    g = PETSc.ScalarType(- 9.81)
    solve(mesh, k, t_end, num_time_steps, T_i, f_T,
          f_u, g, materials, material_mt, bcs, bc_mt)


if __name__ == "__main__":
    main()
