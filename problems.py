# TODO Add nicer problem specification interface.

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


class Problem():
    def __init__(self):
        self.t = 0

    def T(self, x):
        return np.sin(np.pi * x[0]) * np.cos(np.pi * x[1]) * \
            np.sin(np.pi * self.t)

    def f(self, x):
        # Dependence on material parameters has been explicitly computed
        return np.pi * ((np.sin(x[0] * np.pi)**2 * np.sin(np.pi * self.t)**2 *
                         np.cos(x[1] * np.pi)**2 + 1.3) *
                        (np.sin(x[0] * np.pi)**2 * np.sin(np.pi * self.t)**2
                         * np.cos(x[1] * np.pi)**2 + 2.7) *
                        np.cos(np.pi * self.t) + 2 * np.pi *
                        (np.sin(x[0] * np.pi)**2 * np.sin(np.pi * self.t)**2 *
                        np.cos(x[1] * np.pi)**2 + 4.1) * np.sin(np.pi * self.t)
                        - 2 * np.pi * np.sin(x[0] * np.pi)**2 * np.sin(x[1] *
                        np.pi)**2 * np.sin(np.pi * self.t)**3 - 2 * np.pi *
                        np.sin(np.pi * self.t)**3 * np.cos(x[0] * np.pi)**2 *
                        np.cos(x[1] * np.pi)**2) * np.sin(x[0] * np.pi) * \
            np.cos(x[1] * np.pi)

    def materials(self, mesh):
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

        # FIXME This code is duplicated for setting BC's. Make function
        regions = [lambda x: x[0] <= 0.5,
                   lambda x: x[0] >= 0.5]
        cell_indices, cell_markers = [], []
        tdim = mesh.topology.dim
        # Use index in the `regions` list as the unique marker
        for marker, locator in enumerate(regions):
            cells = locate_entities(mesh, tdim, locator)
            cell_indices.append(cells)
            cell_markers.append(np.full(len(cells), marker))
        cell_indices = np.array(np.hstack(cell_indices), dtype=np.int32)
        cell_markers = np.array(np.hstack(cell_markers), dtype=np.int32)
        sorted_cells = np.argsort(cell_indices)
        mt = MeshTags(mesh, tdim, cell_indices[sorted_cells],
                      cell_markers[sorted_cells])
        return materials, mt

    def bcs(self, mesh):
        boundaries = [lambda x: np.isclose(x[0], 0),
                      lambda x: np.isclose(x[0], 1),
                      lambda x: np.isclose(x[1], 0),
                      lambda x: np.isclose(x[1], 1)]

        def neumann_bc(x):
            # NOTE This is just the Neumann BC for the right boundary
            # TODO Implement with UFL instead?
            return np.pi * (np.sin(x[0] * np.pi)**2 *
                            np.sin(np.pi * self.t)**2 *
                            np.cos(x[1] * np.pi)**2 + 4.1) * \
                np.sin(np.pi * self.t) * np.cos(x[0] * np.pi) \
                * np.cos(x[1] * np.pi)

        # Robin BC
        def T_inf(x):
            # NOTE This is just the Robin BC (T_inf) for the left boundary
            return ((np.sin(x[0] * np.pi)**2 * np.sin(np.pi * self.t)**2 *
                     np.cos(x[1] * np.pi)**2 + 3.5) * np.sin(x[0] * np.pi)
                    - np.pi * (np.sin(x[0] * np.pi)**2 *
                    np.sin(np.pi * self.t)**2 * np.cos(x[1] * np.pi)**2 +
                    4.1) * np.cos(x[0] * np.pi)) * np.sin(np.pi * self.t) * \
                np.cos(x[1] * np.pi) / \
                (np.sin(x[0] * np.pi)**2 * np.sin(np.pi * self.t)**2 *
                    np.cos(x[1] * np.pi)**2 + 3.5)

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
        bcs = [{"type": "robin",
                "value": T_inf,
                "h": h},
               {"type": "neumann",
                "value": neumann_bc},
               {"type": "dirichlet",
                "value": self.T},
               {"type": "dirichlet",
                "value": self.T}]

        facet_indices, facet_markers = [], []
        fdim = mesh.topology.dim - 1
        # Use index in the `boundaries` list as the unique marker
        for marker, locator in enumerate(boundaries):
            facets = locate_entities_boundary(mesh, fdim, locator)
            facet_indices.append(facets)
            facet_markers.append(np.full(len(facets), marker))
        facet_indices = np.array(np.hstack(facet_indices), dtype=np.int32)
        facet_markers = np.array(np.hstack(facet_markers), dtype=np.int32)
        sorted_facets = np.argsort(facet_indices)
        mt = MeshTags(mesh, fdim, facet_indices[sorted_facets],
                      facet_markers[sorted_facets])
        return bcs, mt
