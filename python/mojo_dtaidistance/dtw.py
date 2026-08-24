"""Dynamic Time Warping with the public API of :mod:`dtaidistance.dtw`."""

from __future__ import annotations

import array
import math
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from ._core import (
    DTWSettings,
    call_values,
    inner_code,
    pack_series,
    pair_arrays,
)
from ._lib import addr, f64, i64, lib

inf = float("inf")
_DISTANCE_WORKERS = min(36, os.cpu_count() or 1)
_DISTANCE_POOL = ThreadPoolExecutor(max_workers=_DISTANCE_WORKERS)


def distance(s1, s2, **kwargs) -> float:
    settings = DTWSettings.for_dtw(s1, s2, **kwargs)
    a, b, ndim = pair_arrays(s1, s2, settings.use_ndim)
    if settings.max_length_diff is not None and abs(len(a) - len(b)) > settings.max_length_diff:
        return inf
    if not len(a) or not len(b):
        return inf
    work = np.empty(2 * (len(b) + 1), dtype=np.float64)
    return float(
        lib().mdd_distance(
            addr(a), addr(b), len(a), len(b), ndim,
            *call_values(settings, a, b, ndim), addr(work)
        )
    )


def distance_fast(s1, s2, use_pruning=True, **kwargs):
    kwargs["use_pruning"] = use_pruning
    return distance(s1, s2, **kwargs)


def warping_paths(s1, s2, psi_neg=True, keep_int_repr=False, **kwargs):
    settings = DTWSettings.for_dtw(s1, s2, **kwargs)
    a, b, ndim = pair_arrays(s1, s2, settings.use_ndim)
    if settings.max_length_diff is not None and abs(len(a) - len(b)) > settings.max_length_diff:
        return inf, None
    if not len(a) or not len(b):
        paths = np.full((len(a) + 1, len(b) + 1), inf, dtype=np.float64)
        paths[0, 0] = 0.0
        return (0.0 if not len(a) and not len(b) else inf), paths
    paths = np.empty((len(a) + 1, len(b) + 1), dtype=np.float64)
    d = lib().mdd_paths(
        addr(a), addr(b), len(a), len(b), ndim,
        *call_values(settings, a, b, ndim),
        int(keep_int_repr), int(psi_neg), addr(paths),
    )
    return float(d), paths


def warping_paths_fast(
    s1, s2, psi_neg=True, keep_int_repr=False, compact=False, **kwargs
):
    if compact:
        raise NotImplementedError("compact warping-path matrices are not implemented")
    return warping_paths(
        s1, s2, psi_neg=psi_neg, keep_int_repr=keep_int_repr, **kwargs
    )


def warping_path(from_s, to_s, include_distance=False, use_ndim=False, **kwargs):
    d, paths = warping_paths(from_s, to_s, use_ndim=use_ndim, **kwargs)
    path = best_path(paths)
    return (path, d) if include_distance else path


def warping_path_fast(
    from_s, to_s, include_distance=False, use_lowmem=False, **kwargs
):
    if use_lowmem:
        raise NotImplementedError("the Hirschberg low-memory path is not implemented")
    use_ndim = np.asarray(from_s).ndim > 1
    return warping_path(
        from_s, to_s, include_distance=include_distance, use_ndim=use_ndim, **kwargs
    )


def best_path(paths, row=None, col=None, use_max=False, penalty=0):
    i = paths.shape[0] - 1 if row is None else row
    j = paths.shape[1] - 1 if col is None else col
    path = []
    if paths[i, j] != -1:
        path.append((i - 1, j - 1))
    choose = np.argmax if use_max else np.argmin
    while i > 0 and j > 0:
        move = choose(
            [
                paths[i - 1, j - 1],
                paths[i - 1, j] + penalty,
                paths[i, j - 1] + penalty,
            ]
        )
        if move == 0:
            i, j = i - 1, j - 1
        elif move == 1:
            i -= 1
        else:
            j -= 1
        if paths[i, j] != -1:
            path.append((i - 1, j - 1))
    if path:
        path.pop()
    path.reverse()
    return path


def best_path2(paths):
    return best_path(paths)


def warping_amount(path):
    return sum(
        path[i - 1][0] + 1 != path[i][0]
        or path[i - 1][1] + 1 != path[i][1]
        for i in range(1, len(path))
    )


def warp(from_s, to_s, path=None, **kwargs):
    if path is None:
        path = warping_path(from_s, to_s, **kwargs)
    warped = array.array("d", [0.0] * len(to_s))
    counts = array.array("i", [0] * len(to_s))
    for source, target in path:
        warped[target] += from_s[source]
        counts[target] += 1
    for i in range(len(to_s)):
        warped[i] /= counts[i]
    return warped, path


def warping_path_penalty(s1, s2, penalty_post=0, **kwargs):
    d, paths = warping_paths(s1, s2, **kwargs)
    path = best_path(paths)
    steps = []
    for i in range(1, len(path)):
        if path[i - 1][0] + 1 != path[i][0] or path[i - 1][1] + 1 != path[i][1]:
            d += penalty_post
        steps.append(
            paths[path[i][0] + 1, path[i][1] + 1]
            - paths[path[i - 1][0] + 1, path[i - 1][1] + 1]
        )
    return [d, path, steps, paths]


def lb_keogh(s1, s2, **kwargs):
    settings = DTWSettings(**kwargs)
    a, b, ndim = pair_arrays(s1, s2, settings.use_ndim)
    if not len(a) or not len(b):
        return inf
    return float(
        lib().mdd_lb_keogh(
            addr(a), addr(b), len(a), len(b), ndim,
            int(settings.window or 0), inner_code(settings.inner_dist)
        )
    )


def ub_euclidean(s1, s2, inner_dist="squared euclidean"):
    use_ndim = np.asarray(s1).ndim > 1
    a, b, ndim = pair_arrays(s1, s2, use_ndim)
    if not len(a) and not len(b):
        return 0.0
    if not len(a) or not len(b):
        raise IndexError("Euclidean upper bound is undefined for one empty series")
    return float(
        lib().mdd_euclidean(
            addr(a), addr(b), len(a), len(b), ndim, inner_code(inner_dist)
        )
    )


def _complete_block(block, count):
    if block is None or block == 0:
        return ((0, count), (0, count)), True
    return block, not (len(block) > 2 and block[2] is False)


def _pairs(block, count):
    block, triu = _complete_block(block, count)
    result = []
    for r in range(block[0][0], block[0][1]):
        start = max(r + 1, block[1][0]) if triu else block[1][0]
        for c in range(start, min(count, block[1][1])):
            result.append((r, c))
    return result


def distance_matrix(
    s, block=None, compact=False, parallel=False, use_mp=False,
    show_progress=False, only_triu=False, **kwargs
):
    del show_progress
    use_parallel = bool(parallel or use_mp)
    settings = DTWSettings(**kwargs)
    data, offsets, arrays, ndim = pack_series(s, settings.use_ndim)
    count = len(arrays)
    if block is not None and len(block) > 2 and block[2] is False and not compact:
        raise Exception("Block cannot have a third argument triu=false with compact=false")
    pairs_list = _pairs(block, count)
    result = np.empty(len(pairs_list), dtype=np.float64)
    if pairs_list:
        pairs = i64(pairs_list)
        max_len = max(len(a) for a in arrays)
        work_size = 2 * (max_len + 1)
        work_stride = (work_size + 7) // 8 * 8
        parallel_batch = (
            use_parallel
            and len(pairs_list) >= 32
            and len(pairs_list) * work_stride >= 8192
        )
        work_slots = min(len(pairs_list), _DISTANCE_WORKERS) if parallel_batch else 1
        work = np.empty(work_slots * work_stride, dtype=np.float64)
        psi = settings.split_psi()
        args = (
            addr(data), addr(offsets), ndim, int(settings.window or 0),
            float(settings.max_dist or 0.0), float(settings.max_step or 0.0),
            float(settings.penalty or 0.0), *psi, inner_code(settings.inner_dist),
        )

        def run_chunk(worker):
            first = worker * len(pairs_list) // work_slots
            last = (worker + 1) * len(pairs_list) // work_slots
            lib().mdd_distance_pairs(
                args[0], args[1], addr(pairs[first:last]), last - first,
                *args[2:], 0, work_stride,
                addr(work[worker * work_stride:]), addr(result[first:last]),
            )

        if parallel_batch:
            list(_DISTANCE_POOL.map(run_chunk, range(work_slots)))
        else:
            run_chunk(0)
        if settings.max_length_diff is not None:
            for i, (aidx, bidx) in enumerate(pairs_list):
                if abs(len(arrays[aidx]) - len(arrays[bidx])) > settings.max_length_diff:
                    result[i] = inf
    if compact:
        return result
    return distances_array_to_matrix(result, count, block=block, only_triu=only_triu)


def distance_matrix_fast(
    s, max_dist=None, use_pruning=True, max_length_diff=None, window=None,
    max_step=None, penalty=None, psi=None, block=None, compact=False,
    parallel=True, use_mp=False, only_triu=False,
    inner_dist="squared euclidean", use_c=True
):
    return distance_matrix(
        s, block=block, compact=compact, parallel=parallel, use_mp=use_mp,
        only_triu=only_triu, max_dist=max_dist, use_pruning=use_pruning,
        max_length_diff=max_length_diff, window=window, max_step=max_step,
        penalty=penalty, psi=psi, inner_dist=inner_dist, use_c=use_c
    )


def distances_array_to_matrix(dists, nb_series, block=None, only_triu=False):
    matrix = np.full((nb_series, nb_series), inf, dtype=np.float64)
    pairs = _pairs(block, nb_series)
    if pairs:
        rows, cols = zip(*pairs)
        matrix[rows, cols] = dists
        if not only_triu:
            matrix[cols, rows] = dists
    if not only_triu:
        np.fill_diagonal(matrix, 0.0)
    return matrix


def distance_array_index(a, b, nb_series):
    if a == b:
        raise ValueError("Distance between the same series is not available.")
    if a > b:
        a, b = b, a
    return sum(nb_series - r - 1 for r in range(a)) + b - a - 1
