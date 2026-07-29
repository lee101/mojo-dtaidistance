# mojo-dtaidistance

The compute-heavy DTW distance and clustering core of
[dtaidistance](https://github.com/wannesm/dtaidistance), implemented in
[Mojo](https://www.modular.com/mojo) and exposed to Python with the upstream
names and signatures.

```python
import numpy as np
from mojo_dtaidistance import dtw, dtw_barycenter
from mojo_dtaidistance.clustering import KMeans

series = np.array([
    [0., 0, 1, 2, 1, 0],
    [0., 1, 2, 1, 0, 0],
    [5., 5, 6, 7, 6, 5],
    [5., 6, 7, 6, 5, 5],
])

print(dtw.distance_fast(series[0], series[1]))
print(dtw.distance_matrix_fast(series, parallel=False))
print(dtw_barycenter.dba_loop(series[:2], max_it=5))

np.random.seed(0)
clusters, iterations = KMeans(
    2, show_progress=False
).fit(series, use_parallel=False)
print(clusters, iterations)
```

For the covered subset, migration is an import change from `dtaidistance` to
`mojo_dtaidistance`. NumPy arrays and ordinary Python sequences are accepted.

## Coverage

| upstream module | covered API |
| --- | --- |
| `dtw` | `DTWSettings`, `distance`, `distance_fast`, `warping_paths`, `warping_paths_fast`, `warping_path`, `warping_path_fast`, `best_path`, `best_path2`, `warp`, `warping_amount`, `warping_path_penalty` |
| `dtw` matrices and bounds | `distance_matrix`, `distance_matrix_fast`, ragged series, blocks, compact distance arrays, `only_triu`, `distances_array_to_matrix`, `distance_array_index`, `lb_keogh`, `ub_euclidean` |
| `dtw_ndim` | dependent multivariate distance, warping paths, distance matrices, Euclidean upper bound |
| `ed` | `distance`, `distance_fast` |
| `dtw_barycenter` | `dba`, `dba_loop`, masks, scalar and multivariate series |
| `clustering` | DBA `KMeans` including k-means++ initialization; PAM `KMedoids` |

Distance settings support `window`, `max_dist`, `max_step`,
`max_length_diff`, `penalty`, scalar or four-part `psi`, pruning, and the
`squared euclidean` and `euclidean` inner distances. The test suite checks
numerical and behavioral parity against the real `dtaidistance` 2.4.0 package,
including ragged and multivariate inputs, full path matrices, DBA, seeded
K-means means and partitions, and k-medoids partitions.

Not covered:

- affinity/similarity DTW, probabilistic paths and probabilistic DBA;
- compact *warping-path* matrices and the Hirschberg low-memory path variant;
- custom Python inner-distance callables;
- OpenMP and multiprocessing backends; parallel matrices use Mojo CPU tasks;
- hierarchical clustering, subsequence search, visualization and preprocessing.

`KMedoids` uses an in-repo PAM implementation instead of requiring
pyclustering at runtime. Plot methods and upstream-specific serialization are
not implemented.

## Install

```bash
pixi install
pixi run build
pixi run test
pixi run bench
```

`pixi install` provides the pinned Mojo nightly, Python, NumPy, the upstream
package used by the parity tests, and pyclustering for the upstream k-medoids
comparison. The build task produces:

```text
dist/libmojo-dtaidistance.so
```

The Python package also rebuilds a missing or stale library on first use. Set
`MOJO_DTAIDISTANCE_LIB=/absolute/path/to/libmojo-dtaidistance.so` to use a
prebuilt library.

## Performance

Measured with `pixi run bench` against dtaidistance 2.4.0 on an Intel Xeon
E5-2697 v4 at 2.30 GHz, 72 logical CPUs, x86-64. Times are the best of three
warm runs.

| case | mojo-dtaidistance | dtaidistance | result |
| --- | ---: | ---: | ---: |
| distance (4096 x 4096) | 48.50 ms | 96.94 ms | 2.00x faster |
| distance window=32 (50k x 50k) | 8.24 ms | 17.14 ms | 2.08x faster |
| DTW_D (1200 x 1200 x 4) | 8.46 ms | 11.12 ms | 1.31x faster |
| warping_paths (1500 x 1500) | 24.00 ms | 23.90 ms | 1.00x slower |
| distance_matrix serial (64 x 256) | 384.01 ms | 771.58 ms | 2.01x faster |
| distance_matrix parallel (64 x 256) | 48.56 ms | 51.76 ms | 1.07x faster |
| DBA one update (48 x 256) | 20.48 ms | 27.65 ms | 1.35x faster |
| KMeans k=3, 3 it (45 x 192) | 105.89 ms | 283.54 ms | 2.68x faster |

The common squared-Euclidean distance kernel overwrites each active row
directly instead of clearing it first. SIMD handles bulk initialization and
Euclidean reductions, including scalar remainder elements. Large independent
distance batches use bounded CPU parallelism with private, cache-padded scratch
rows; small batches stay serial to avoid launch overhead. Contiguous NumPy
batches remain zero-copy across the FFI boundary.

There is no GPU path. DTW's left/up/diagonal recurrence limits parallelism, so
this port concentrates on CPU kernels.

## How it works

`src/capi.mojo` is one compilation unit containing the numerics and
`@export(... )` C ABI wrappers. Python owns all input, output and scratch
memory. ctypes passes contiguous buffers as integer addresses; the exported
wrappers reconstruct `UnsafePointer[Float64, AnyOrigin[mut=True]]` values
inside Mojo.

Scalar series are contiguous `float64` values. Multivariate series use
row-major `(length, ndim)` storage. Ragged collections are concatenated into
one data buffer with an `int64` point-offset table. Distance matrices,
cluster assignment and DBA each cross the FFI once for the whole operation.
The Mojo library does not allocate or retain memory.

The distance kernel uses two caller-owned rows and limits both computation and
scratch clearing to the Sakoe-Chiba band. Full warping paths use an
`(n + 1) x (m + 1)` row-major accumulated-cost matrix. DBA fills that matrix
once per selected series and accumulates traceback associations directly in
Mojo instead of returning every path to Python.

## License

MIT
