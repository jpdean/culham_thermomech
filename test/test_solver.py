import thermomech
from mpi4py import MPI
from dolfinx.mesh import create_unit_square
from dolfinx import fem
import ufl
import numpy as np
from problems import (TimeDependentExpression, create_mesh_tags,
                      ufl_poly_from_table_data, compute_error_L2_norm,
                      compute_convergence_rate)
from petsc4py import PETSc


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


def u(x):
    return ufl.as_vector((ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1]),
                          ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])))


def c(T):
    # Dummy data representing 1.3 + T**2
    x = np.array([0.0, 0.25, 0.50, 0.75, 1.0])
    y = np.array([1.3, 1.3625, 1.55, 1.8625, 2.3])
    return ufl_poly_from_table_data(x, y, 2, T)


def rho(T):
    # Dummy data representing 2.7 + T**2
    x = np.array([0.0, 0.25, 0.50, 0.75, 1.0])
    y = np.array([2.7, 2.7625, 2.95, 3.2625, 3.7])
    return ufl_poly_from_table_data(x, y, 2, T)


def kappa(T):
    # Dummy data representing 4.1 + T**2
    x = np.array([0.0, 0.25, 0.50, 0.75, 1.0])
    y = np.array([4.1, 4.1625, 4.35, 4.6625, 5.1])
    return ufl_poly_from_table_data(x, y, 2, T)


# Create two materials (they are the same, just mat_1
# is a numpy polynomial fit of mat_2)
materials = []
# TODO Test ufl_poly_from_table_data for elastic properties
materials.append({"name": "mat_1",
                  "c": c,
                  "rho": rho,
                  "kappa": kappa,
                  "nu": 0.33,
                  "E": lambda T: 1.0 + 0.1 * T**2,
                  "thermal_strain": (lambda T: 0.1 + 0.01 * T**3,
                                     1.5)})
materials.append({"name": "mat_2",
                  "c": lambda T: 1.3 + T**2,
                  "rho": lambda T: 2.7 + T**2,
                  "kappa": lambda T: 4.1 + T**2,
                  "nu": 0.33,
                  "E": lambda T: 1.0 + 0.1 * T**2,
                  "thermal_strain": (lambda T: 0.1 + 0.01 * T**3,
                                     1.5)})


def get_material_mt(mesh):
    regions = [lambda x: x[0] <= 0.5,
               lambda x: x[0] >= 0.5]
    return create_mesh_tags(mesh, regions, mesh.topology.dim)


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
    # Test ufl.conditional works OK for complicated coefficients
    # which should be approximated with multiple polynomials.
    # Dummy data representing 2.7 + T**2
    x = np.array([0.0, 0.25, 0.50, 0.75, 1.0])
    y = np.array([3.5, 3.5625, 3.75, 4.0625, 4.5])
    h_poly = ufl_poly_from_table_data(x, y, 2, T)
    # NOTE For this problem, this will always be false as the solution
    # is zero on this boundary
    return ufl.conditional(T > 0.5, 3.5 + T**2, h_poly)


# Think of nicer way to deal with Robin bc
# TODO Change "value" to expression
# TODO Add pressure BC
bcs = [{"type": "convection",
        "value": T_inf,
        "h": h},
       {"type": "heat_flux",
        "value": neumann_bc},
       {"type": "temperature",
        "value": T_expr},
       {"type": "temperature",
        "value": T_expr},
       {"type": "displacement",
        "value": np.array([0, 0], dtype=PETSc.ScalarType)}]


def get_bc_mt(mesh):
    boundaries = [lambda x: np.isclose(x[0], 0),
                  lambda x: np.isclose(x[0], 1),
                  lambda x: np.isclose(x[1], 0),
                  lambda x: np.isclose(x[1], 1),
                  lambda x: np.logical_or(
                      np.logical_or(np.isclose(x[0], 0), np.isclose(x[0], 1)),
                      np.logical_or(np.isclose(x[1], 0), np.isclose(x[1], 1)))]
    return create_mesh_tags(mesh, boundaries, mesh.topology.dim - 1)


def test_temporal_convergence():
    t_end = 1.5
    n = 64
    k = 1
    num_time_steps = [16, 32]
    mesh = create_unit_square(MPI.COMM_WORLD, n, n)
    # TODO Compute
    f_u = fem.Constant(mesh, np.array([0, 0], dtype=PETSc.ScalarType))
    errors_L2 = []
    V_e = fem.FunctionSpace(mesh, ("Lagrange", k + 3))
    T_e = fem.Function(V_e)
    material_mt = get_material_mt(mesh)
    bc_mt = get_bc_mt(mesh)
    for i in range(len(num_time_steps)):
        T_expr.t = 0
        f_T_expr.t = 0
        for bc in bcs:
            if isinstance(bc["value"], TimeDependentExpression):
                bc["value"].t = 0
        (T_h, u_h) = thermomech.solve(mesh, k, t_end, num_time_steps[i],
                                      T_expr, f_T_expr, f_u, materials,
                                      material_mt, bcs, bc_mt)
        T_e.interpolate(T_expr)
        errors_L2.append(compute_error_L2_norm(mesh.comm, T_h, T_e))

    r = compute_convergence_rate(errors_L2, num_time_steps)

    assert(np.isclose(r, 1.0, atol=0.1))


def test_spatial_convergence():
    t_end = 1.5
    num_time_steps = 200
    k = 1
    errors_L2 = []
    ns = [8, 16]

    for i in range(len(ns)):
        T_expr.t = 0
        f_T_expr.t = 0
        for bc in bcs:
            if isinstance(bc["value"], TimeDependentExpression):
                bc["value"].t = 0
        # TODO Use refine rather than create new mesh?
        mesh = create_unit_square(MPI.COMM_WORLD, ns[i], ns[i])
        material_mt = get_material_mt(mesh)
        bc_mt = get_bc_mt(mesh)
        # TODO Compute
        f_u = fem.Constant(mesh, np.array([0, 0], dtype=PETSc.ScalarType))
        (T_h, u_h) = thermomech.solve(mesh, k, t_end, num_time_steps,
                                      T_expr, f_T_expr, f_u, materials,
                                      material_mt, bcs, bc_mt)

        V_e = fem.FunctionSpace(mesh, ("Lagrange", k + 3))
        T_e = fem.Function(V_e)
        T_e.interpolate(T_expr)

        errors_L2.append(compute_error_L2_norm(mesh.comm, T_h, T_e))

    r = compute_convergence_rate(errors_L2, ns)

    assert(np.isclose(r, 2.0, atol=0.1))
