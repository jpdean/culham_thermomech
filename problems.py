# TODO Add nicer problem specification interface.

from mpi4py import MPI
from dolfinx.mesh import create_unit_square
import numpy as np
from dolfinx.mesh import locate_entities_boundary, MeshTags, locate_entities
import ufl


def ufl_poly_from_table_data(x, y, degree, u):
    """Given a list of point data x and y, this function returns a fitted
    polynomial of degree `degree` in terms of the UFL `Function` `u`"""
    coeffs = np.polynomial.Polynomial.fit(x, y, degree).convert().coef

    poly = 0
    for n in range(degree + 1):
        poly += coeffs[n] * u**n
    return poly


class TimeDependentExpression():
    def __init__(self, expression):
        self.t = 0
        self.expression = expression

    def __call__(self, x):
        return self.expression(x, self.t)


def create_problem_0(mesh):
    T = TimeDependentExpression(lambda x, t:
                                np.sin(np.pi * x[0]) *
                                np.cos(np.pi * x[1]) *
                                np.sin(np.pi * t))
    f = TimeDependentExpression(
        lambda x, t: np.pi * ((np.sin(x[0] * np.pi)**2 * np.sin(np.pi * t)**2 *
                               np.cos(x[1] * np.pi)**2 + 1.3) *
                              (np.sin(x[0] * np.pi)**2 * np.sin(np.pi * t)**2
                               * np.cos(x[1] * np.pi)**2 + 2.7) *
                              np.cos(np.pi * t) + 2 * np.pi *
                              (np.sin(x[0] * np.pi)**2 * np.sin(np.pi * t)**2 *
                               np.cos(x[1] * np.pi)**2 + 4.1) * np.sin(np.pi * t)
                              - 2 * np.pi * np.sin(x[0] * np.pi)**2 * np.sin(x[1] *
                                                                             np.pi)**2 * np.sin(np.pi * t)**3 - 2 * np.pi *
                              np.sin(np.pi * t)**3 * np.cos(x[0] * np.pi)**2 *
                              np.cos(x[1] * np.pi)**2) * np.sin(x[0] * np.pi) *
        np.cos(x[1] * np.pi))

    # Materials
    # Specific heat capacity
    def c(T):
        # Dummy data representing 1.3 + T**2
        x = np.array([0.0, 0.25, 0.50, 0.75, 1.0])
        y = np.array([1.3, 1.3625, 1.55, 1.8625, 2.3])
        return ufl_poly_from_table_data(x, y, 2, T)

    # Density
    def rho(T):
        # Dummy data representing 2.7 + T**2
        x = np.array([0.0, 0.25, 0.50, 0.75, 1.0])
        y = np.array([2.7, 2.7625, 2.95, 3.2625, 3.7])
        return ufl_poly_from_table_data(x, y, 2, T)

    # Thermal conductivity
    def kappa(T):
        # Dummy data representing 4.1 + T**2
        x = np.array([0.0, 0.25, 0.50, 0.75, 1.0])
        y = np.array([4.1, 4.1625, 4.35, 4.6625, 5.1])
        return ufl_poly_from_table_data(x, y, 2, T)

    # Create two materials (they are the same, just mat_1
    # is a numpy polynomial fit of mat_2)
    materials = []
    mat_1 = {"name": "mat_1",
             "c": c,
             "rho": rho,
             "kappa": kappa}
    mat_2 = {"name": "mat_1",
             "c": lambda T: 1.3 + T**2,
             "rho": lambda T: 2.7 + T**2,
             "kappa": lambda T: 4.1 + T**2}
    materials.append(mat_1)
    materials.append(mat_2)

    def create_mesh_tags(regions, edim):
        entity_indices, entity_markers = [], []
        # Use index in the `regions` list as the unique marker
        for marker, locator in enumerate(regions):
            # TODO Use locate_entities_boundary for boundaries?
            entities = locate_entities(mesh, edim, locator)
            entity_indices.append(entities)
            entity_markers.append(np.full(len(entities), marker))
        entity_indices = np.array(np.hstack(entity_indices), dtype=np.int32)
        entity_markers = np.array(np.hstack(entity_markers), dtype=np.int32)
        sorted_entities = np.argsort(entity_indices)
        mt = MeshTags(mesh, edim, entity_indices[sorted_entities],
                      entity_markers[sorted_entities])
        return mt

    regions = [lambda x: x[0] <= 0.5,
               lambda x: x[0] >= 0.5]
    tdim = mesh.topology.dim
    material_mt = create_mesh_tags(regions, tdim)

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
                                 np.sin(np.pi * t)**2 * np.cos(x[1] * np.pi)**2 +
                                 4.1) * np.cos(x[0] * np.pi)) * np.sin(np.pi * t) *
        np.cos(x[1] * np.pi) /
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
    bcs = [{"type": "robin",
            "value": T_inf,
            "h": h},
           {"type": "neumann",
            "value": neumann_bc},
           {"type": "dirichlet",
            "value": T},
           {"type": "dirichlet",
            "value": T}]

    boundaries = [lambda x: np.isclose(x[0], 0),
                  lambda x: np.isclose(x[0], 1),
                  lambda x: np.isclose(x[1], 0),
                  lambda x: np.isclose(x[1], 1)]

    bc_mt = create_mesh_tags(boundaries, tdim - 1)

    return T, f, materials, material_mt, bcs, bc_mt
