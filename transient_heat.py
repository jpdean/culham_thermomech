import numpy as np

from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import fem
from dolfinx.mesh import create_unit_square
from dolfinx.io import XDMFFile
from dolfinx.nls import NewtonSolver

import ufl

from problems import Problem


def solve(mesh, k, t_end, num_time_steps, problem, use_iterative_solver=True):
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

    F = - ufl.inner(delta_t * f, v) * ufl.dx

    materials, mat_mt = problem.materials(mesh)
    dx = ufl.Measure("dx", domain=mesh, subdomain_data=mat_mt)

    # FIXME I think this creates a new kernel for every material
    for marker, mat in enumerate(materials):
        c = mat["c"](T_h)
        rho = mat["rho"](T_h)
        kappa = mat["kappa"](T_h)
        F += ufl.inner(rho * c * T_h, v) * dx(marker) + \
            delta_t * ufl.inner(kappa * ufl.grad(T_h),
                                ufl.grad(v)) * dx(marker) - \
            ufl.inner(rho * c * T_n, v) * dx(marker)

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
        elif bc_type == "robin":
            T_inf = bc_funcs[marker]
            h = bc["h"](T_h)
            F += delta_t * ufl.inner(h * (T_h - T_inf), v) * ds(marker)
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
        problem.t += delta_t.value

        # FIXME Only update time dependent functions
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
