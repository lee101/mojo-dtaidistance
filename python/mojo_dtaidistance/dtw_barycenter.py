"""Dynamic Barycenter Averaging backed by a fused Mojo update."""

from __future__ import annotations

import random

import numpy as np

from . import dtw
from ._core import DTWSettings, inner_code, pack_series
from ._lib import addr, f64, lib


def _initial_center(series, mask, nb_initial_samples, kwargs):
    selected = np.flatnonzero(mask)
    if nb_initial_samples is None:
        return series[int(selected[0])].copy()
    sample_count = min(int(nb_initial_samples), len(selected))
    sample_idx = random.sample(list(selected), sample_count)
    sample = [series[i] for i in sample_idx]
    matrix = dtw.distance_matrix(sample, **kwargs)
    return sample[int(np.argmin(matrix.sum(axis=1)))].copy()


def dba(s, c, mask=None, samples=None, use_c=False, nb_initial_samples=None, **kwargs):
    del use_c
    if samples is not None and samples > 0:
        raise NotImplementedError("probabilistic DBA sampling is not implemented")
    guessed_ndim = bool(kwargs.get("use_ndim", False))
    if not guessed_ndim:
        first = np.asarray(s[0] if not isinstance(s, np.ndarray) else s[0])
        guessed_ndim = first.ndim == 2
    kwargs["use_ndim"] = guessed_ndim
    settings = DTWSettings(**kwargs)
    data, offsets, series, ndim = pack_series(s, settings.use_ndim)
    if not series:
        raise ValueError("DBA requires at least one series")
    if mask is None:
        selected = np.ones(len(series), dtype=np.uint8)
    else:
        mask_array = np.asarray(mask)
        if mask_array.shape != (len(series),):
            raise ValueError("mask must contain one value per series")
        selected = np.ascontiguousarray(mask_array, dtype=np.bool_).view(np.uint8)
    if not selected.any():
        shape = (len(series[0]), ndim) if ndim > 1 else (len(series[0]),)
        return np.zeros(shape, dtype=np.float64)
    if c is None:
        c = _initial_center(series, selected.astype(bool), nb_initial_samples, kwargs)
    center = f64(c, copy=True)
    if ndim == 1:
        if center.ndim != 1:
            raise ValueError("center must be one-dimensional")
    elif center.ndim != 2 or center.shape[1] != ndim:
        raise ValueError("center dimensions do not match the series")
    center_len = len(center)
    max_len = max(len(item) for item in series)
    paths = np.empty((center_len + 1) * (max_len + 1), dtype=np.float64)
    sums = np.zeros(center_len * ndim, dtype=np.float64)
    counts = np.zeros(center_len, dtype=np.float64)
    psi = settings.split_psi()
    lib().mdd_dba_update(
        addr(data), addr(offsets), len(series), addr(center), center_len, ndim,
        addr(selected), int(settings.window or 0), float(settings.max_dist or 0.0),
        float(settings.max_step or 0.0), float(settings.penalty or 0.0),
        *psi, inner_code(settings.inner_dist), addr(paths), addr(sums), addr(counts)
    )
    result = sums.reshape(center_len, ndim)
    original = center.reshape(center_len, ndim)
    nonempty = counts > 0
    result[nonempty] /= counts[nonempty, None]
    result[~nonempty] = original[~nonempty]
    return result[:, 0].copy() if ndim == 1 else result.copy()


def dba_loop(
    s, c=None, max_it=10, thr=0.001, mask=None, keep_averages=False,
    use_c=False, nb_initial_samples=None, nb_prob_samples=None, **kwargs
):
    if nb_prob_samples not in (None, 0):
        raise NotImplementedError("probabilistic DBA sampling is not implemented")
    items = [np.asarray(item, dtype=np.float64) for item in s]
    if not items:
        raise ValueError("DBA requires at least one series")
    selected = (
        np.ones(len(items), dtype=bool)
        if mask is None else np.asarray(mask, dtype=bool)
    )
    if selected.shape != (len(items),):
        raise ValueError("mask must contain one value per series")
    if c is None:
        c = _initial_center(items, selected, nb_initial_samples, kwargs)
    current = np.array(c, dtype=np.float64, copy=True)
    averages = []
    for _ in range(max_it):
        updated = dba(items, current, mask=selected, use_c=use_c, **kwargs)
        if keep_averages:
            averages.append(updated.copy())
        if current.ndim == 1:
            diff = np.mean(np.abs(updated - current))
        else:
            diff = np.mean(np.max(np.abs(updated - current), axis=1))
        current = updated
        if thr is not None and diff <= thr:
            break
    return (current, averages) if keep_averages else current
