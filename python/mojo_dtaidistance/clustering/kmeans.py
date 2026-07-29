"""DTW K-means with Mojo assignment and DBA kernels."""

from __future__ import annotations

import math
import random

import numpy as np

from .. import dtw
from .._core import DTWSettings, inner_code, pack_series
from .._lib import addr, lib
from ..dtw_barycenter import dba_loop
from .medoids import KMedoids, Medoids


class KMeans(Medoids):
    def __init__(
        self, k, max_it=10, max_dba_it=10, thr=0.0001, drop_stddev=None,
        nb_prob_samples=None, dists_options=None, show_progress=True,
        initialize_with_kmedoids=False, initialize_with_kmeanspp=True,
        initialize_sample_size=None
    ):
        self.means = [None] * k
        self.max_it = max_it
        self.max_dba_it = max_dba_it
        self.thr = thr
        self.drop_stddev = drop_stddev
        self.initialize_with_kmeanspp = initialize_with_kmeanspp
        self.initialize_with_kmedoids = initialize_with_kmedoids
        self.initialize_sample_size = initialize_sample_size
        self.nb_prob_samples = nb_prob_samples
        super().__init__(None, dict(dists_options or {}), k, show_progress)

    def kmeansplusplus_centers(self, series):
        series = list(series)
        n_samples = (
            min(2 + int(math.log(self.k)), len(series) - self.k)
            if self.initialize_sample_size is None
            else self.initialize_sample_size
        )
        n_samples = max(1, min(n_samples, len(series)))
        idx = np.random.randint(0, len(series))
        minimum = np.square(
            dtw.distance_matrix(
                series, block=((idx, idx + 1), (0, len(series)), False),
                compact=True, **self.dists_options
            )
        )
        indices = [idx]
        candidates = np.empty((n_samples, len(series)))
        for _ in range(1, self.k):
            total = minimum.sum()
            weights = None if total == 0 else minimum / total
            trial_idx = np.random.choice(
                len(series), size=n_samples, replace=False, p=weights
            )
            for row, candidate in enumerate(trial_idx):
                dists = dtw.distance_matrix(
                    series,
                    block=((candidate, candidate + 1), (0, len(series)), False),
                    compact=True,
                    **self.dists_options,
                )
                candidates[row] = np.minimum(np.square(dists), minimum)
            best = int(np.argmin(candidates.sum(axis=1)))
            idx = int(trial_idx[best])
            minimum = candidates[best].copy()
            indices.append(idx)
        return [np.array(series[i], dtype=np.float64, copy=True) for i in indices]

    def kmedoids_centers(self, series):
        sample_size = min(
            self.initialize_sample_size or self.k * 20, len(series)
        )
        indices = np.random.choice(len(series), sample_size, replace=False)
        sample = [series[i] for i in indices]
        model = KMedoids(
            dtw.distance_matrix, {**self.dists_options, "compact": False}, k=self.k
        )
        clusters = model.fit(sample)
        return [np.array(sample[i], copy=True) for i in clusters]

    def _assign(self, packed, offsets, arrays, ndim):
        center_data, center_offsets, centers, center_ndim = pack_series(
            self.means, use_ndim=ndim > 1
        )
        if center_ndim != ndim:
            raise ValueError("center dimensions do not match series")
        settings = DTWSettings(
            **{k: v for k, v in self.dists_options.items() if k != "compact"}
        )
        labels = np.empty(len(arrays), dtype=np.int64)
        distances = np.empty(len(arrays), dtype=np.float64)
        work = np.empty(
            2 * (max(len(center) for center in centers) + 1), dtype=np.float64
        )
        lib().mdd_assign(
            addr(packed), addr(offsets), len(arrays), addr(center_data),
            addr(center_offsets), self.k, ndim, int(settings.window or 0),
            float(settings.max_dist or 0.0), float(settings.max_step or 0.0),
            float(settings.penalty or 0.0), *settings.split_psi(),
            inner_code(settings.inner_dist), addr(work), addr(labels), addr(distances)
        )
        return labels, distances

    def fit_fast(self, series, monitor_distances=None):
        return self.fit(series, use_parallel=True, monitor_distances=monitor_distances)

    def fit(self, series, use_parallel=True, monitor_distances=None):
        del use_parallel
        raw = list(series)
        if self.k < 1 or self.k > len(raw):
            raise ValueError("k must be between 1 and the number of series")
        use_ndim = np.asarray(raw[0]).ndim == 2
        packed, offsets, arrays, ndim = pack_series(raw, use_ndim)
        self.series = arrays
        if self.initialize_with_kmeanspp:
            self.means = self.kmeansplusplus_centers(arrays)
        elif self.initialize_with_kmedoids:
            self.means = self.kmedoids_centers(arrays)
        else:
            indices = np.random.choice(len(arrays), self.k, replace=False)
            self.means = [arrays[int(i)].copy() for i in indices]
        previous_mask = np.zeros((self.k, len(arrays)), dtype=bool)
        performed_it = 1
        drop = max(self.drop_stddev, 4) if self.drop_stddev is not None else None
        for _ in range(self.max_it):
            performed_it += 1
            labels, distances = self._assign(packed, offsets, arrays, ndim)
            assignments = list(zip(labels.tolist(), distances.tolist()))
            if monitor_distances is not None and monitor_distances(assignments, False) is False:
                break
            mask = np.zeros_like(previous_mask)
            if self.drop_stddev not in (None, 0):
                limits = np.empty(self.k)
                for cluster in range(self.k):
                    values = distances[labels == cluster]
                    limits[cluster] = (
                        values.mean() + values.std() * drop if len(values) else 0.0
                    )
                drop = (drop + self.drop_stddev) / 2
                for idx, cluster in enumerate(labels):
                    mask[cluster, idx] = distances[idx] <= limits[cluster]
            else:
                mask[labels, np.arange(len(arrays))] = True
            if np.array_equal(mask, previous_mask):
                break
            previous_mask = mask.copy()
            for cluster in range(self.k):
                if not mask[cluster].any():
                    idx = int(np.argmax(distances))
                    mask[:, idx] = False
                    mask[cluster, idx] = True
                    distances[idx] = 0.0
            old_means = self.means
            means = []
            for cluster in range(self.k):
                members = np.flatnonzero(mask[cluster])
                seed = int(members[np.argmin(distances[members])])
                options = {
                    k: v for k, v in self.dists_options.items()
                    if k not in ("use_c", "compact")
                }
                options["use_ndim"] = ndim > 1
                means.append(
                    dba_loop(
                        arrays, c=arrays[seed], mask=mask[cluster],
                        max_it=self.max_dba_it, thr=self.thr,
                        nb_prob_samples=self.nb_prob_samples, **options
                    )
                )
            change = sum(
                np.abs(new - old).max(axis=-1).sum()
                if new.ndim > 1 else np.abs(new - old).sum()
                for new, old in zip(means, old_means)
            ) / sum(len(mean) for mean in means)
            self.means = means
            if change <= self.thr:
                break
        labels, distances = self._assign(packed, offsets, arrays, ndim)
        assignments = list(zip(labels.tolist(), distances.tolist()))
        if monitor_distances is not None:
            monitor_distances(assignments, True)
        self.cluster_idx = {
            cluster: set(np.flatnonzero(labels == cluster).tolist())
            for cluster in range(self.k)
        }
        return self.cluster_idx, performed_it
