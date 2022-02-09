# TODO Add nicer problem specification interface.

import numpy as np
from dolfinx.mesh import MeshTags, locate_entities
from dolfinx import fem
import ufl
from mpi4py import MPI


def ufl_linear_interp(xs, ys, u):
    ms = []
    cs = []
    for i in range(len(xs) - 1):
        ms.append((ys[i + 1] - ys[i]) / (xs[i + 1] - xs[i]))
        cs.append(ys[i] - ms[i] * xs[i])

    conditions = [ufl.gt(u, x) for x in xs]
    pieces = [m * u + c for (m, c) in zip(ms, cs)]
    # If u < xs[-1], extrapolate using the last piece
    pieces.append(pieces[-1])

    interp = ufl.conditional(conditions[1], pieces[1], pieces[0])
    for i in range(1, len(conditions) - 1):
        interp = ufl.conditional(conditions[i + 1],
                                 pieces[i + 1],
                                 interp)
    return interp


def ufl_poly_from_table_data(x, y, u, degree):
    """Given a list of point data x and y, this function returns a fitted
    polynomial of degree `degree` in terms of the UFL `Function` `u`"""
    coeffs = np.polynomial.Polynomial.fit(x, y, degree).convert().coef
    poly = 0
    for n in range(degree + 1):
        poly += coeffs[n] * u**n
    return poly


def create_mesh_tags_from_locators(mesh, locators, edim):
    entity_indices, entity_markers = [], []
    # Use index in the `regions` list as the unique marker
    for marker, locator in enumerate(locators):
        # TODO Use locate_entities_boundary for boundaries?
        entities = locate_entities(mesh, edim, locator)
        entity_indices.append(entities)
        entity_markers.append(np.full(len(entities), marker))
    entity_indices = np.array(np.hstack(entity_indices), dtype=np.int32)
    # Check no entities are duplicated
    assert(len(np.unique(entity_indices)) == len(entity_indices))
    entity_markers = np.array(np.hstack(entity_markers), dtype=np.int32)
    sorted_entities = np.argsort(entity_indices)
    mt = MeshTags(mesh, edim, entity_indices[sorted_entities],
                  entity_markers[sorted_entities])
    return mt


class TimeDependentExpression():
    def __init__(self, expression):
        self.t = 0
        self.expression = expression

    def __call__(self, x):
        return self.expression(x, self.t)


def compute_error_L2_norm(comm, v_h, v):
    return np.sqrt(comm.allreduce(
        fem.assemble_scalar(fem.form((v_h - v)**2 * ufl.dx)), op=MPI.SUM))


def compute_convergence_rate(errors_L2, ns):
    return np.log(errors_L2[-1] / errors_L2[-2]) / \
        np.log(ns[-2] / ns[-1])
