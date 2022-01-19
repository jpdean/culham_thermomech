import transient_heat
from mpi4py import MPI
from dolfinx.mesh import create_unit_square
from dolfinx import fem
import ufl
import numpy as np
from problems import create_problem_0


def compute_error_L2_norm(comm, T_h, T_e):
    return np.sqrt(comm.allreduce(
        fem.assemble_scalar(fem.form((T_h - T_e)**2 * ufl.dx)), op=MPI.SUM))


def compute_convergence_rate(errors_L2, ns):
    return np.log(errors_L2[-1] / errors_L2[-2]) / \
        np.log(ns[-2] / ns[-1])


def test_temporal_convergence():
    t_end = 1.5
    n = 64
    k = 1
    num_time_steps = [16, 32]
    mesh = create_unit_square(MPI.COMM_WORLD, n, n)
    errors_L2 = []
    V_e = fem.FunctionSpace(mesh, ("Lagrange", k + 3))
    T_e = fem.Function(V_e)
    for i in range(len(num_time_steps)):
        T, f, materials, material_mt, bcs, bc_mt = create_problem_0(mesh)
        T_h = transient_heat.solve(mesh, k, t_end, num_time_steps[i], T, f,
                                   materials, material_mt, bcs, bc_mt)
        T_e.interpolate(T)
        errors_L2.append(compute_error_L2_norm(mesh.comm, T_h, T_e))

    r = compute_convergence_rate(errors_L2, num_time_steps)

    assert(np.isclose(r, 1.0, atol=0.1))


# def test_spatial_convergence():
#     t_end = 1.5
#     num_time_steps = 200
#     k = 1
#     problem = Problem()
#     errors_L2 = []
#     ns = [8, 16]

#     for i in range(len(ns)):
#         problem.t = 0
#         # TODO Use refine rather than create new mesh?
#         mesh = create_unit_square(MPI.COMM_WORLD, ns[i], ns[i])
#         T_h = transient_heat.solve(mesh, k, t_end, num_time_steps, problem)

#         V_e = fem.FunctionSpace(mesh, ("Lagrange", k + 3))
#         T_e = fem.Function(V_e)
#         T_e.interpolate(problem.T)

#         errors_L2.append(compute_error_L2_norm(mesh.comm, T_h, T_e))

#     r = compute_convergence_rate(errors_L2, ns)

#     assert(np.isclose(r, 2.0, atol=0.1))
