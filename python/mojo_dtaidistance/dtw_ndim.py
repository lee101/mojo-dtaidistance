"""Multivariate aliases matching :mod:`dtaidistance.dtw_ndim`."""

from __future__ import annotations

from . import dtw


def distance(
    s1, s2, window=None, max_dist=None, max_step=None, max_length_diff=None,
    penalty=None, psi=None, use_c=False, use_pruning=False,
    inner_dist="squared euclidean"
):
    return dtw.distance(
        s1, s2, window=window, max_dist=max_dist, max_step=max_step,
        max_length_diff=max_length_diff, penalty=penalty, psi=psi,
        use_c=use_c, use_pruning=use_pruning, inner_dist=inner_dist,
        use_ndim=True
    )


def distance_fast(
    s1, s2, window=None, max_dist=None, max_step=None, max_length_diff=None,
    penalty=None, psi=None, use_pruning=False, inner_dist="squared euclidean"
):
    return distance(
        s1, s2, window=window, max_dist=max_dist, max_step=max_step,
        max_length_diff=max_length_diff, penalty=penalty, psi=psi,
        use_pruning=use_pruning, inner_dist=inner_dist
    )


def warping_paths(*args, **kwargs):
    return dtw.warping_paths(*args, use_ndim=True, **kwargs)


def warping_paths_fast(*args, **kwargs):
    return dtw.warping_paths_fast(*args, use_ndim=True, **kwargs)


def warping_path(*args, **kwargs):
    return dtw.warping_path(*args, use_ndim=True, **kwargs)


def warping_path_fast(*args, **kwargs):
    return dtw.warping_path_fast(*args, **kwargs)


def distance_matrix(
    s, ndim=None, max_dist=None, use_pruning=False, max_length_diff=None,
    window=None, max_step=None, penalty=None, psi=None, block=None,
    compact=False, parallel=False, use_c=False, use_mp=False,
    show_progress=False, only_triu=False, inner_dist="squared euclidean"
):
    del ndim
    return dtw.distance_matrix(
        s, max_dist=max_dist, use_pruning=use_pruning,
        max_length_diff=max_length_diff, window=window, max_step=max_step,
        penalty=penalty, psi=psi, block=block, compact=compact,
        parallel=parallel, use_c=use_c, use_mp=use_mp,
        show_progress=show_progress, only_triu=only_triu,
        inner_dist=inner_dist, use_ndim=True
    )


def distance_matrix_fast(
    s, ndim=None, max_dist=None, max_length_diff=None, window=None,
    max_step=None, penalty=None, psi=None, block=None, compact=False,
    parallel=True, only_triu=False, inner_dist="squared euclidean", use_c=True
):
    del ndim
    return dtw.distance_matrix(
        s, max_dist=max_dist, use_pruning=True,
        max_length_diff=max_length_diff, window=window, max_step=max_step,
        penalty=penalty, psi=psi, block=block, compact=compact,
        parallel=parallel, use_c=use_c, only_triu=only_triu,
        inner_dist=inner_dist, use_ndim=True
    )


def ub_euclidean(s1, s2, inner_dist="squared euclidean"):
    return dtw.ub_euclidean(s1, s2, inner_dist=inner_dist)
