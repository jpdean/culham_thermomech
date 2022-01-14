import transient_heat
from mpi4py import MPI
from dolfinx.mesh import create_unit_square
from dolfinx import fem
import ufl
import numpy as np


def compute_error_L2_norm(comm, T_h, T_e):
    return np.sqrt(comm.allreduce(
        fem.assemble_scalar(fem.form((T_h - T_e)**2 * ufl.dx)), op=MPI.SUM))


def test_temporal_convergence():
    t_end = 1.5
    n = 64
    k = 1
    mesh = create_unit_square(MPI.COMM_WORLD, n, n)
    problem = transient_heat.Problem()
    num_time_steps = [16, 32]

    errors_L2 = []
    V_e = fem.FunctionSpace(mesh, ("Lagrange", k + 3))
    T_e = fem.Function(V_e)
    problem.t = t_end
    T_e.interpolate(problem.T)
    for i in range(len(num_time_steps)):
        problem.t = 0
        T_h = transient_heat.solve(mesh, k, t_end, num_time_steps[i], problem)
        errors_L2.append(compute_error_L2_norm(mesh.comm, T_h, T_e))

    r = np.log(errors_L2[-1] / errors_L2[-2]) / \
        np.log(num_time_steps[-2] / num_time_steps[-1])

    assert(np.isclose(r, 1.0, atol=0.1))


def test_spatial_convergence():
    t_end = 1.5
    num_time_steps = 2000
    n = 8
    k = 1
    mesh = create_unit_square(MPI.COMM_WORLD, n, n)
    problem = transient_heat.Problem()
    T_h = transient_heat.solve(mesh, k, t_end, num_time_steps, problem)

    V_e = fem.FunctionSpace(mesh, ("Lagrange", k + 3))
    T_e = fem.Function(V_e)
    T_e.interpolate(problem.T)

    errors_L2 = []
    errors_L2.append(compute_error_L2_norm(mesh.comm, T_h, T_e))

    mesh = create_unit_square(MPI.COMM_WORLD, 2 * n, 2 * n)
    problem.t = 0
    T_h = transient_heat.solve(mesh, k, t_end, num_time_steps, problem)

    V_e = fem.FunctionSpace(mesh, ("Lagrange", k + 3))
    T_e = fem.Function(V_e)
    T_e.interpolate(problem.T)

    errors_L2.append(compute_error_L2_norm(mesh.comm, T_h, T_e))

    # Check doubling number of elements reduces error by factor of four
    assert(np.isclose(errors_L2[1] / errors_L2[0], 0.25, atol=0.015))
