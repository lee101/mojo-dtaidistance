"""Benchmarks against the native upstream dtaidistance package."""

from __future__ import annotations

import gc
import math
import os
import platform
import random
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))

import dtaidistance
from dtaidistance import dtw as upstream_dtw
from dtaidistance import dtw_barycenter as upstream_barycenter
from dtaidistance import dtw_ndim as upstream_ndim
from dtaidistance.clustering import KMeans as UpstreamKMeans

from mojo_dtaidistance import dtw, dtw_barycenter, dtw_ndim
from mojo_dtaidistance.clustering import KMeans


def timeit(fn, repeat=3):
    best = math.inf
    for _ in range(repeat):
        gc.collect()
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best


def cpu_name():
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


rng = np.random.default_rng(2026)

a_long = np.ascontiguousarray(rng.normal(size=4096))
b_long = np.ascontiguousarray(rng.normal(size=4096))
a_band = np.ascontiguousarray(rng.normal(size=50_000))
b_band = np.ascontiguousarray(rng.normal(size=50_000))
a_ndim = np.ascontiguousarray(rng.normal(size=(1200, 4)))
b_ndim = np.ascontiguousarray(rng.normal(size=(1200, 4)))
a_paths = np.ascontiguousarray(rng.normal(size=1500))
b_paths = np.ascontiguousarray(rng.normal(size=1500))
matrix_series = np.ascontiguousarray(rng.normal(size=(64, 256)))
dba_series = np.ascontiguousarray(rng.normal(size=(48, 256)))

base = np.sin(np.linspace(0, 4, 192))
kmeans_series = np.ascontiguousarray(
    np.array(
        [base + rng.normal(0, 0.08, 192) for _ in range(15)]
        + [np.roll(base, 35) + 2 + rng.normal(0, 0.08, 192) for _ in range(15)]
        + [base[::-1] - 2 + rng.normal(0, 0.08, 192) for _ in range(15)]
    )
)


def mojo_kmeans():
    np.random.seed(7)
    random.seed(7)
    return KMeans(
        3, max_it=3, max_dba_it=2, show_progress=False
    ).fit(kmeans_series, use_parallel=False)


def upstream_kmeans():
    np.random.seed(7)
    random.seed(7)
    return UpstreamKMeans(
        3, max_it=3, max_dba_it=2, show_progress=False,
        dists_options={"use_c": True}
    ).fit(kmeans_series, use_parallel=False)


CASES = [
    (
        "distance (4096 x 4096)",
        lambda: dtw.distance_fast(a_long, b_long, use_pruning=False),
        lambda: upstream_dtw.distance_fast(a_long, b_long, use_pruning=False),
    ),
    (
        "distance window=32 (50k x 50k)",
        lambda: dtw.distance_fast(a_band, b_band, window=32, use_pruning=False),
        lambda: upstream_dtw.distance_fast(
            a_band, b_band, window=32, use_pruning=False
        ),
    ),
    (
        "DTW_D (1200 x 1200 x 4)",
        lambda: dtw_ndim.distance_fast(a_ndim, b_ndim, use_pruning=False),
        lambda: upstream_ndim.distance_fast(a_ndim, b_ndim, use_pruning=False),
    ),
    (
        "warping_paths (1500 x 1500)",
        lambda: dtw.warping_paths_fast(a_paths, b_paths),
        lambda: upstream_dtw.warping_paths_fast(a_paths, b_paths),
    ),
    (
        "distance_matrix serial (64 x 256)",
        lambda: dtw.distance_matrix_fast(
            matrix_series, parallel=False, use_pruning=False
        ),
        lambda: upstream_dtw.distance_matrix_fast(
            matrix_series, parallel=False, use_pruning=False
        ),
    ),
    (
        "distance_matrix parallel (64 x 256)",
        lambda: dtw.distance_matrix_fast(
            matrix_series, parallel=True, use_pruning=False
        ),
        lambda: upstream_dtw.distance_matrix_fast(
            matrix_series, parallel=True, use_pruning=False
        ),
    ),
    (
        "DBA one update (48 x 256)",
        lambda: dtw_barycenter.dba(dba_series, dba_series[0]),
        lambda: upstream_barycenter.dba(
            dba_series, dba_series[0], use_c=True
        ),
    ),
    (
        "KMeans k=3, 3 it (45 x 192)",
        mojo_kmeans,
        upstream_kmeans,
    ),
]


def main():
    dtw.distance_fast(a_long[:8], b_long[:8])
    print(
        f"Machine: {cpu_name()}, {os.cpu_count()} logical CPUs, "
        f"{platform.machine()}"
    )
    print(f"Upstream: dtaidistance {dtaidistance.__version__}")
    print()
    print("| case | mojo-dtaidistance | dtaidistance | result |")
    print("| --- | ---: | ---: | ---: |")
    for name, mojo_fn, upstream_fn in CASES:
        mojo_value = mojo_fn()
        upstream_value = upstream_fn()
        if name.startswith("distance ") or name.startswith("DTW_D"):
            if not np.allclose(mojo_value, upstream_value):
                raise RuntimeError(f"benchmark parity failed for {name}")
        mojo_time = timeit(mojo_fn)
        upstream_time = timeit(upstream_fn)
        ratio = upstream_time / mojo_time
        result = (
            f"{ratio:.2f}x faster"
            if ratio >= 1
            else f"{1 / ratio:.2f}x slower"
        )
        print(
            f"| {name} | {mojo_time * 1e3:.2f} ms | "
            f"{upstream_time * 1e3:.2f} ms | {result} |"
        )


if __name__ == "__main__":
    main()
