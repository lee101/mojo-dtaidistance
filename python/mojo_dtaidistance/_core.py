"""Shared validation and container packing for the public modules."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._lib import addr, f64, i64, lib


def inner_code(inner_dist) -> int:
    if inner_dist in (None, "squared euclidean", "squared_euclidean"):
        return 0
    if inner_dist == "euclidean":
        return 1
    raise ValueError(
        "inner_dist must be 'squared euclidean' or 'euclidean'; "
        "custom Python inner distances cannot run in Mojo"
    )


def split_psi(psi) -> tuple[int, int, int, int]:
    if psi is None:
        return 0, 0, 0, 0
    if isinstance(psi, (int, np.integer)):
        values = (int(psi),) * 4
    elif isinstance(psi, (tuple, list)) and len(psi) == 4:
        if any(not isinstance(v, (int, np.integer)) for v in psi):
            raise TypeError("psi values must be integers")
        values = tuple(int(v) for v in psi)
    else:
        raise ValueError("psi must be an integer or a 4-tuple")
    if any(v < 0 for v in values):
        raise ValueError("psi values must be non-negative")
    return values


@dataclass
class DTWSettings:
    window: int | None = None
    use_pruning: bool = False
    max_dist: float | None = None
    max_step: float | None = None
    max_length_diff: int | None = None
    penalty: float | None = None
    psi: int | tuple[int, int, int, int] | None = None
    inner_dist: str = "squared euclidean"
    use_ndim: bool = False
    use_c: bool = False

    def __post_init__(self):
        inner_code(self.inner_dist)
        split_psi(self.psi)
        for name in ("window", "max_length_diff"):
            value = getattr(self, name)
            if value is not None:
                if not isinstance(value, (int, np.integer)):
                    raise TypeError(f"{name} must be an integer")
                if value < 0:
                    raise ValueError(f"{name} must be non-negative")
        for name in ("max_dist", "max_step", "penalty"):
            value = getattr(self, name)
            if value is not None and (not np.isfinite(value) or value < 0):
                raise ValueError(f"{name} must be finite and non-negative")

    @staticmethod
    def wrap(settings) -> "DTWSettings":
        if isinstance(settings, DTWSettings):
            return settings
        if settings is None:
            return DTWSettings()
        if isinstance(settings, dict) and "dtw_settings" in settings:
            if len(settings) != 1:
                raise ValueError("dtw_settings cannot be combined with other arguments")
            return DTWSettings.wrap(settings["dtw_settings"])
        if isinstance(settings, dict):
            return DTWSettings(**settings)
        raise TypeError("settings must be a DTWSettings or dict")

    @staticmethod
    def for_dtw(s1, s2, **kwargs) -> "DTWSettings":
        return DTWSettings.wrap(kwargs)

    def split_psi(self):
        return split_psi(self.psi)

    def kwargs(self):
        return {
            "window": self.window,
            "use_pruning": self.use_pruning,
            "max_dist": self.max_dist,
            "max_step": self.max_step,
            "max_length_diff": self.max_length_diff,
            "penalty": self.penalty,
            "psi": self.psi,
            "inner_dist": self.inner_dist,
            "use_ndim": self.use_ndim,
            "use_c": self.use_c,
        }

    def c_kwargs(self):
        return {
            "window": self.window or 0,
            "max_dist": self.max_dist or 0,
            "max_step": self.max_step or 0,
            "max_length_diff": self.max_length_diff or 0,
            "penalty": self.penalty or 0,
            "psi": self.psi or 0,
            "use_pruning": int(self.use_pruning),
            "inner_dist": inner_code(self.inner_dist),
        }


def pair_arrays(s1, s2, use_ndim: bool = False):
    a = f64(s1)
    b = f64(s2)
    if use_ndim:
        if a.ndim != 2 or b.ndim != 2:
            raise ValueError("multivariate series must have shape (length, ndim)")
        if a.shape[1] != b.shape[1]:
            raise ValueError("series have different numbers of dimensions")
        ndim = int(a.shape[1])
    else:
        if a.ndim != 1 or b.ndim != 1:
            raise ValueError("one-dimensional DTW expects one-dimensional series")
        ndim = 1
    return a, b, ndim


def pack_series(series, use_ndim: bool = False):
    if isinstance(series, np.ndarray):
        packed = f64(series)
        if use_ndim and series.ndim == 3:
            items = [packed[i] for i in range(len(packed))]
        elif not use_ndim and series.ndim == 2:
            items = [packed[i] for i in range(len(packed))]
        else:
            items = list(series)
    else:
        items = list(series)
    if not items:
        return np.empty(0, np.float64), i64([0]), [], 1
    arrays = [f64(item) for item in items]
    if use_ndim:
        if any(a.ndim != 2 for a in arrays):
            raise ValueError("multivariate series must have shape (length, ndim)")
        ndim = int(arrays[0].shape[1])
        if ndim == 0 or any(a.shape[1] != ndim for a in arrays):
            raise ValueError("all series must have the same number of dimensions")
    else:
        if any(a.ndim != 1 for a in arrays):
            raise ValueError("each series must be one-dimensional")
        ndim = 1
    lengths = [len(a) for a in arrays]
    offsets = i64(np.r_[0, np.cumsum(lengths, dtype=np.int64)])
    if isinstance(series, np.ndarray) and (
        (use_ndim and packed.ndim == 3) or (not use_ndim and packed.ndim == 2)
    ):
        data = packed.reshape(-1)
    else:
        data = np.ascontiguousarray(
            np.concatenate([a.reshape(-1) for a in arrays]), dtype=np.float64
        )
    return data, offsets, arrays, ndim


def call_values(settings: DTWSettings, a, b, ndim: int):
    max_dist = float(settings.max_dist or 0.0)
    if settings.use_pruning and max_dist == 0.0 and len(a) and len(b):
        max_dist = lib().mdd_euclidean(
            addr(a), addr(b), len(a), len(b), ndim, inner_code(settings.inner_dist)
        )
    return (
        int(settings.window or 0),
        max_dist,
        float(settings.max_step or 0.0),
        float(settings.penalty or 0.0),
        *settings.split_psi(),
        inner_code(settings.inner_dist),
    )
