"""Euclidean upper-bound helpers."""

from .dtw import ub_euclidean


def distance(s1, s2, inner_dist="squared euclidean", use_ndim=False):
    del use_ndim
    return ub_euclidean(s1, s2, inner_dist=inner_dist)


distance_fast = distance
