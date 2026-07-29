"""K-medoids clustering over a DTW distance matrix."""

from __future__ import annotations

import random

import numpy as np


class Medoids:
    def __init__(self, dists_fun, dists_options, k, show_progress=True):
        self.dists_fun = dists_fun
        self.dists_options = dists_options
        self.show_progress = show_progress
        self.k = k
        self.series = None
        self.cluster_idx = None

    def plot(self, *args, **kwargs):
        raise NotImplementedError("clustering visualization is outside this port's scope")


class KMedoids(Medoids):
    def __init__(
        self, dists_fun, dists_options, k=None, initial_medoids=None,
        show_progress=True
    ):
        options = dict(dists_options)
        options["compact"] = False
        self.initial_medoids = initial_medoids
        if k is None:
            if initial_medoids is None:
                raise AttributeError("Both k and initial_medoids cannot be None")
            k = len(initial_medoids)
        elif initial_medoids is not None and k != len(initial_medoids):
            raise AttributeError(
                "The length of initial_medoids and k should be identical"
            )
        super().__init__(dists_fun, options, k, show_progress)

    def fit(self, series):
        self.series = list(series)
        if self.k < 1 or self.k > len(self.series):
            raise ValueError("k must be between 1 and the number of series")
        distances = np.asarray(
            self.dists_fun(self.series, **self.dists_options), dtype=np.float64
        )
        distances = np.minimum(distances, distances.T)
        np.fill_diagonal(distances, 0.0)
        medoids = (
            list(self.initial_medoids)
            if self.initial_medoids is not None
            else random.sample(range(len(self.series)), k=self.k)
        )
        labels = np.argmin(distances[:, medoids], axis=1)
        cost = distances[np.arange(len(self.series)), np.asarray(medoids)[labels]].sum()
        while True:
            best = None
            best_cost = cost
            nonmedoids = [i for i in range(len(self.series)) if i not in medoids]
            for slot in range(self.k):
                for candidate in nonmedoids:
                    trial = medoids.copy()
                    trial[slot] = candidate
                    trial_cost = np.min(distances[:, trial], axis=1).sum()
                    if trial_cost < best_cost - 1e-12:
                        best_cost = trial_cost
                        best = (slot, candidate)
            if best is None:
                break
            medoids[best[0]] = best[1]
            cost = best_cost
        labels = np.argmin(distances[:, medoids], axis=1)
        self.initial_medoids = medoids
        self.cluster_idx = {
            medoid: set(np.flatnonzero(labels == cluster).tolist())
            for cluster, medoid in enumerate(medoids)
        }
        return self.cluster_idx
