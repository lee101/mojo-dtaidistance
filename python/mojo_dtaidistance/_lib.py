"""ctypes bindings for the compiled Mojo kernels."""

from __future__ import annotations

import ctypes
import os
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "src", "capi.mojo")
LIB = os.environ.get("MOJO_DTAIDISTANCE_LIB") or os.path.join(
    ROOT, "dist", "libmojo-dtaidistance.so"
)

I = ctypes.c_int64
F = ctypes.c_double

_SIGNATURES = {
    "mdd_distance": ([I] * 6 + [F] * 3 + [I] * 6, F),
    "mdd_paths": ([I] * 6 + [F] * 3 + [I] * 8, F),
    "mdd_lb_keogh": ([I] * 7, F),
    "mdd_euclidean": ([I] * 6, F),
    "mdd_distance_pairs": ([I] * 6 + [F] * 3 + [I] * 9, None),
    "mdd_assign": ([I] * 8 + [F] * 3 + [I] * 8, None),
    "mdd_dba_update": ([I] * 8 + [F] * 3 + [I] * 8, None),
}


class BuildError(RuntimeError):
    pass


def build(force: bool = False) -> str:
    if os.environ.get("MOJO_DTAIDISTANCE_LIB"):
        if os.path.exists(LIB):
            return LIB
        raise BuildError(f"MOJO_DTAIDISTANCE_LIB does not exist: {LIB}")
    stale = not os.path.exists(LIB) or os.path.getmtime(LIB) < os.path.getmtime(SRC)
    if force or stale:
        proc = subprocess.run(
            ["bash", os.path.join(ROOT, "build", "build.sh")],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=1800,
        )
        if proc.returncode != 0 or not os.path.exists(LIB):
            raise BuildError((proc.stderr or proc.stdout).strip()[:4000])
    return LIB


_library: ctypes.CDLL | None = None


def lib() -> ctypes.CDLL:
    global _library
    if _library is None:
        _library = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            fn = getattr(_library, name)
            fn.argtypes = argtypes
            fn.restype = restype
    return _library


def f64(values, *, copy: bool = False) -> np.ndarray:
    source = np.asarray(values)
    if source.dtype.kind in "cSUVO":
        raise TypeError("series values must be real numbers representable as float64")
    converted = np.asarray(source, dtype=np.float64, order="C")
    if source.dtype.kind in "iu" and not np.array_equal(
        converted.astype(source.dtype), source
    ):
        raise ValueError("integer series values cannot be represented exactly as float64")
    if source.dtype.kind == "f" and source.dtype.itemsize > 8:
        finite = np.isfinite(source)
        if not np.array_equal(converted.astype(source.dtype)[finite], source[finite]):
            raise ValueError("series values would be narrowed when converted to float64")
    if copy:
        return np.array(converted, dtype=np.float64, order="C", copy=True)
    return np.ascontiguousarray(converted)


def i64(values) -> np.ndarray:
    return np.ascontiguousarray(values, dtype=np.int64)


def addr(values: np.ndarray) -> int:
    address = int(values.ctypes.data)
    if address == 0:
        raise ValueError("cannot pass a null NumPy buffer across the FFI boundary")
    return address
