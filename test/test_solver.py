import thermomech
from mpi4py import MPI
from dolfinx.mesh import create_unit_square
from dolfinx import fem
import ufl
import numpy as np
from utils import (TimeDependentExpression, create_mesh_tags_from_locators,
                   ufl_poly_from_table_data, compute_error_L2_norm,
                   compute_convergence_rate)
from petsc4py import PETSc


# TODO Test ufl_poly_from_table_data and ufl_linear_interp

T_expr = TimeDependentExpression(lambda x, t:
                                 np.sin(np.pi * x[0]) *
                                 np.cos(np.pi * x[1]) *
                                 np.sin(np.pi * t))

f_T_expr = TimeDependentExpression(
    lambda x, t: np.pi * ((np.sin(x[0] * np.pi)**2 * np.sin(np.pi * t)**2 *
                           np.cos(x[1] * np.pi)**2 + 1.3) *
                          (np.sin(x[0] * np.pi)**2 * np.sin(np.pi * t)**2
                           * np.cos(x[1] * np.pi)**2 + 2.7) *
                          np.cos(np.pi * t) + 2 * np.pi *
                          (np.sin(x[0] * np.pi)**2 * np.sin(np.pi * t)**2 *
                           np.cos(x[1] * np.pi)**2 + 4.1) *
                          np.sin(np.pi * t)
                          - 2 * np.pi * np.sin(x[0] * np.pi)**2 *
                          np.sin(x[1] * np.pi)**2 * np.sin(np.pi * t)**3 -
                          2 * np.pi * np.sin(np.pi * t)**3 *
                          np.cos(x[0] * np.pi)**2 *
                          np.cos(x[1] * np.pi)**2) * np.sin(x[0] * np.pi) *
    np.cos(x[1] * np.pi))


def u_expr(x):
    return ufl.as_vector((ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1]),
                          ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])))


def c(T):
    # Dummy data representing 1.3 + T**2
    x = np.array([0.0, 0.25, 0.50, 0.75, 1.0])
    y = np.array([1.3, 1.3625, 1.55, 1.8625, 2.3])
    return ufl_poly_from_table_data(x, y, T, 2)


def rho(T):
    # Dummy data representing 2.7 + T**2
    x = np.array([0.0, 0.25, 0.50, 0.75, 1.0])
    y = np.array([2.7, 2.7625, 2.95, 3.2625, 3.7])
    return ufl_poly_from_table_data(x, y, T, 2)


def kappa(T):
    # Dummy data representing 4.1 + T**2
    x = np.array([0.0, 0.25, 0.50, 0.75, 1.0])
    y = np.array([4.1, 4.1625, 4.35, 4.6625, 5.1])
    return ufl_poly_from_table_data(x, y, T, 2)


volume_ids = {"volume_0": 8,
              "volume_1": 2}
boundary_ids = {}
boundary_ids["T"] = {"boundary_0": 9,
                     "boundary_1": 5,
                     "boundary_2": 6,
                     "boundary_3": 8}
boundary_ids["u"] = {"boundary_0": 1}

# Create two materials (mat_1 is a numpy polynomial fit of mat_2)
materials = {volume_ids["volume_0"]:
             {"name": "mat_1",
              "c": c,
              "rho": rho,
              "kappa": kappa,
              "nu": 0.33,
              "E": lambda T: 1.0 + 0.1 * T**2,
              "thermal_strain": (lambda T: 0.1 + 0.01 * T**3,
                                 1.5)},
             volume_ids["volume_1"]:
             {"name": "mat_2",
              "c": lambda T: 1.3 + T**2,
              "rho": lambda T: 2.7 + T**2,
              "kappa": lambda T: 4.1 + T**2,
              "nu": 0.33,
              "E": lambda T: 1.0 + 0.1 * T**2,
              "thermal_strain": (lambda T: 0.1 + 0.01 * T**3,
                                 1.5)}}


def get_material_mt(mesh):
    regions = {volume_ids["volume_0"]: lambda x: x[0] <= 0.5,
               volume_ids["volume_1"]: lambda x: x[0] >= 0.5}
    return create_mesh_tags_from_locators(mesh, regions, mesh.topology.dim)


# Boundary conditions
neumann_bc = TimeDependentExpression(
    lambda x, t: np.pi * (np.sin(x[0] * np.pi)**2 *
                          np.sin(np.pi * t)**2 *
                          np.cos(x[1] * np.pi)**2 + 4.1) *
    np.sin(np.pi * t) * np.cos(x[0] * np.pi)
    * np.cos(x[1] * np.pi))

T_inf = TimeDependentExpression(
    lambda x, t: ((np.sin(x[0] * np.pi)**2 * np.sin(np.pi * t)**2 *
                   np.cos(x[1] * np.pi)**2 + 3.5) * np.sin(x[0] * np.pi)
                  - np.pi * (np.sin(x[0] * np.pi)**2 *
                             np.sin(np.pi * t)**2 *
                             np.cos(x[1] * np.pi)**2 +
                             4.1) * np.cos(x[0] * np.pi)) *
    np.sin(np.pi * t) * np.cos(x[1] * np.pi) /
    (np.sin(x[0] * np.pi)**2 * np.sin(np.pi * t)**2 *
        np.cos(x[1] * np.pi)**2 + 3.5))


def h(T):
    # Test ufl.conditional
    # Dummy data representing 2.7 + T**2
    x = np.array([0.0, 0.25, 0.50, 0.75, 1.0])
    y = np.array([3.5, 3.5625, 3.75, 4.0625, 4.5])
    h_poly = ufl_poly_from_table_data(x, y, T, 2)
    # NOTE For this problem, this will always be false as the solution
    # is zero on this boundary
    return ufl.conditional(T > 0.5, 3.5 + T**2, h_poly)


# Think of nicer way to deal with Robin bc
# TODO Add pressure BC
bcs = {}
bcs["T"] = {boundary_ids["T"]["boundary_0"]:
            {"type": "convection",
             "value": T_inf,
             "h": h},
            boundary_ids["T"]["boundary_1"]:
            {"type": "heat_flux",
             "value": neumann_bc},
            boundary_ids["T"]["boundary_2"]:
            {"type": "temperature",
             "value": T_expr},
            boundary_ids["T"]["boundary_3"]:
            {"type": "temperature",
             "value": T_expr}}
bcs["u"] = {boundary_ids["u"]["boundary_0"]:
            {"type": "displacement",
             "value": np.array([0, 0], dtype=PETSc.ScalarType)}}


def get_bc_mt(mesh):
    bc_mt = {}
    bc_mt["T"] = create_mesh_tags_from_locators(
        mesh,
        {boundary_ids["T"]["boundary_0"]: lambda x: np.isclose(x[0], 0),
         boundary_ids["T"]["boundary_1"]: lambda x: np.isclose(x[0], 1),
         boundary_ids["T"]["boundary_2"]: lambda x: np.isclose(x[1], 0),
         boundary_ids["T"]["boundary_3"]: lambda x: np.isclose(x[1], 1)},
        mesh.topology.dim - 1)
    bc_mt["u"] = create_mesh_tags_from_locators(
        mesh,
        {boundary_ids["u"]["boundary_0"]: lambda x: np.logical_or(
            np.logical_or(np.isclose(x[0], 0), np.isclose(x[0], 1)),
            np.logical_or(np.isclose(x[1], 0), np.isclose(x[1], 1)))},
        mesh.topology.dim - 1)
    return bc_mt


def compute_f_u(T_expr, t_end, T_e, u_e, materials):
    # The elastic solution doesn't depend on the history of the applied force,
    # so just set the force to be correct at t_end
    T_expr.t = t_end
    T_e.interpolate(T_expr)
    # This problem has two materials for testing, but they have the same
    # properties, so can just use the first
    return - ufl.div(thermomech.sigma(
        u_e, T_e, materials[volume_ids["volume_0"]]["thermal_strain"][1],
        materials[volume_ids["volume_0"]]["thermal_strain"][0](T_e),
        materials[volume_ids["volume_0"]]["E"](T_e),
        materials[volume_ids["volume_0"]]["nu"]))


def test_temporal_convergence():
    t_end = 1.5
    n = 64
    k = 1
    num_time_steps = [8, 16]
    mesh = create_unit_square(MPI.COMM_WORLD, n, n)
    material_mt = get_material_mt(mesh)
    bc_mt = get_bc_mt(mesh)

    errors_L2 = {"T": [], "u": []}
    V_e = fem.FunctionSpace(mesh, ("Lagrange", k + 3))
    T_e = fem.Function(V_e)

    x = ufl.SpatialCoordinate(mesh)
    u_e = u_expr(x)
    f_u = compute_f_u(T_expr, t_end, T_e, u_e, materials)

    g = PETSc.ScalarType(0.0)

    for i in range(len(num_time_steps)):
        T_expr.t = 0
        f_T_expr.t = 0
        for bc in bcs["T"].values():
            if isinstance(bc["value"], TimeDependentExpression):
                bc["value"].t = 0
        delta_t = t_end / num_time_steps[i]
        results = thermomech.solve(mesh, k, delta_t, num_time_steps[i],
                                   T_expr, f_T_expr, f_u, g, materials,
                                   material_mt, bcs, bc_mt)
        errors_L2["T"].append(compute_error_L2_norm(
            mesh.comm, results["T"], T_e))
        errors_L2["u"].append(compute_error_L2_norm(
            mesh.comm, results["u"], u_e))

    r_T = compute_convergence_rate(errors_L2["T"], num_time_steps)
    r_u = compute_convergence_rate(errors_L2["u"], num_time_steps)

    assert (np.isclose(r_T, 1.0, atol=0.1))
    assert (np.isclose(r_u, 1.0, atol=0.1))


def test_spatial_convergence():
    t_end = 1.5
    num_time_steps = 200
    delta_t = t_end / num_time_steps
    k = 1
    errors_L2 = []
    ns = [8, 16]

    g = PETSc.ScalarType(0.0)

    errors_L2 = {"T": [], "u": []}
    for i in range(len(ns)):
        mesh = create_unit_square(MPI.COMM_WORLD, ns[i], ns[i])

        T_expr.t = t_end
        V_e = fem.FunctionSpace(mesh, ("Lagrange", k + 3))
        T_e = fem.Function(V_e)
        T_e.interpolate(T_expr)

        x = ufl.SpatialCoordinate(mesh)
        u_e = u_expr(x)
        f_u = compute_f_u(T_expr, t_end, T_e, u_e, materials)

        T_expr.t = 0
        f_T_expr.t = 0
        for bc in bcs["T"].values():
            if isinstance(bc["value"], TimeDependentExpression):
                bc["value"].t = 0
        # TODO Use refine rather than create new mesh?
        material_mt = get_material_mt(mesh)
        bc_mt = get_bc_mt(mesh)

        results = thermomech.solve(mesh, k, delta_t, num_time_steps,
                                   T_expr, f_T_expr, f_u, g, materials,
                                   material_mt, bcs, bc_mt)

        errors_L2["T"].append(compute_error_L2_norm(
            mesh.comm, results["T"], T_e))
        errors_L2["u"].append(compute_error_L2_norm(
            mesh.comm, results["u"], u_e))

    r_T = compute_convergence_rate(errors_L2["T"], ns)
    r_u = compute_convergence_rate(errors_L2["u"], ns)

    assert (np.isclose(r_T, 2.0, atol=0.1))
    assert (np.isclose(r_u, 2.0, atol=0.1))
