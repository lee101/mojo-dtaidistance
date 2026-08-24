"""Numerical and behavioral parity with dtaidistance 2.x."""

from __future__ import annotations

import random

import numpy as np
import pytest

upstream = pytest.importorskip("dtaidistance")
from dtaidistance import dtw as upstream_dtw
from dtaidistance import dtw_barycenter as upstream_barycenter
from dtaidistance import dtw_ndim as upstream_ndim
from dtaidistance import ed as upstream_ed
from dtaidistance.clustering import KMeans as UpstreamKMeans
from dtaidistance.clustering import KMedoids as UpstreamKMedoids

from mojo_dtaidistance import dtw, dtw_barycenter, dtw_ndim, ed
from mojo_dtaidistance.clustering import KMeans, KMedoids


@pytest.fixture
def pair():
    return (
        np.array([0.0, 0, 1, 2, 1, 0, 1, 0, 0]),
        np.array([0.0, 1, 2, 0, 0, 0, 0, 0, 0]),
    )


def test_published_usage_vector(pair):
    a, b = pair
    assert dtw.distance(a, b) == pytest.approx(upstream_dtw.distance(a, b))
    assert dtw.distance_fast(a, b) == pytest.approx(
        upstream_dtw.distance_fast(a, b)
    )


@pytest.mark.parametrize(
    "options",
    [
        {"window": 1},
        {"window": 3},
        {"penalty": 0.4},
        {"max_step": 0.75},
        {"psi": 2},
        {"psi": (1, 2, 0, 1)},
        {"inner_dist": "euclidean"},
        {"window": 3, "penalty": 0.2, "max_step": 1.5},
    ],
)
def test_distance_options(options):
    rng = np.random.default_rng(8)
    a = rng.normal(size=17)
    b = rng.normal(size=13)
    expected = upstream_dtw.distance(a, b, **options)
    actual = dtw.distance(a, b, **options)
    assert actual == pytest.approx(expected) or (
        np.isinf(actual) and np.isinf(expected)
    )


def test_pruning_and_max_dist(pair):
    a, b = pair
    expected = upstream_dtw.distance_fast(a, b, use_pruning=True)
    assert dtw.distance_fast(a, b, use_pruning=True) == pytest.approx(expected)
    assert np.isinf(dtw.distance(a, b, max_dist=0.5))
    assert np.isinf(upstream_dtw.distance(a, b, max_dist=0.5))


def test_max_length_difference():
    a = np.arange(3.0)
    b = np.arange(8.0)
    assert np.isinf(dtw.distance(a, b, max_length_diff=2))
    assert np.isinf(upstream_dtw.distance(a, b, max_length_diff=2))


@pytest.mark.parametrize("keep_internal", [False, True])
def test_full_warping_paths(pair, keep_internal):
    a, b = pair
    expected_d, expected_paths = upstream_dtw.warping_paths(
        a, b, window=3, penalty=0.25, keep_int_repr=keep_internal
    )
    actual_d, actual_paths = dtw.warping_paths(
        a, b, window=3, penalty=0.25, keep_int_repr=keep_internal
    )
    assert actual_d == pytest.approx(expected_d)
    assert np.allclose(actual_paths, expected_paths)


def test_psi_paths_and_best_path():
    a = np.array([1.0, 2, 3, 4])
    b = np.array([0.0, 1, 2, 3, 4, 8])
    expected_d, expected_paths = upstream_dtw.warping_paths(a, b, psi=(0, 1, 1, 1))
    actual_d, actual_paths = dtw.warping_paths(a, b, psi=(0, 1, 1, 1))
    assert actual_d == pytest.approx(expected_d)
    assert np.allclose(actual_paths, expected_paths)
    assert dtw.best_path(actual_paths) == upstream_dtw.best_path(expected_paths)


def test_psi_cannot_relax_away_entire_alignment():
    a = np.array([0.0])
    b = np.array([0.5])
    expected_d, expected_paths = upstream_dtw.warping_paths(a, b, psi=1)
    actual_d, actual_paths = dtw.warping_paths(a, b, psi=1)
    assert actual_d == pytest.approx(expected_d)
    assert np.allclose(actual_paths, expected_paths)


def test_warping_paths_max_dist_pruning_matrix():
    a = np.array([-1.48817461, 0.21630683])
    b = np.array([0.98437635, -0.54308414])
    expected_d, expected_paths = upstream_dtw.warping_paths(a, b, max_dist=2)
    actual_d, actual_paths = dtw.warping_paths(a, b, max_dist=2)
    assert np.isinf(actual_d) and np.isinf(expected_d)
    assert np.allclose(actual_paths, expected_paths)


def test_path_warp_and_warping_amount(pair):
    a, b = pair
    expected_path, expected_d = upstream_dtw.warping_path(
        a, b, include_distance=True
    )
    actual_path, actual_d = dtw.warping_path(a, b, include_distance=True)
    assert actual_path == expected_path
    assert actual_d == pytest.approx(expected_d)
    assert dtw.warping_amount(actual_path) == upstream_dtw.warping_amount(
        expected_path
    )
    expected_warp, _ = upstream_dtw.warp(a, b, path=expected_path)
    actual_warp, _ = dtw.warp(a, b, path=actual_path)
    assert np.allclose(actual_warp, expected_warp)


def test_bounds(pair):
    a, b = pair
    assert dtw.lb_keogh(a, b, window=3) == pytest.approx(
        upstream_dtw.lb_keogh(a, b, window=3)
    )
    assert dtw.ub_euclidean(a, b) == pytest.approx(
        upstream_dtw.ub_euclidean(a, b)
    )
    assert ed.distance(a, b) == pytest.approx(upstream_ed.distance(a, b))
    assert ed.distance_fast(a, b) == pytest.approx(upstream_ed.distance_fast(a, b))


def test_euclidean_bound_simd_tail():
    rng = np.random.default_rng(31)
    a = rng.normal(size=(5, 3))
    b = rng.normal(size=(5, 3))
    assert dtw.ub_euclidean(a, b) == pytest.approx(np.linalg.norm(a - b))


def test_distance_matrix_equal_length():
    rng = np.random.default_rng(3)
    series = rng.normal(size=(9, 18))
    expected = upstream_dtw.distance_matrix_fast(
        series, parallel=False, window=4, penalty=0.1
    )
    actual = dtw.distance_matrix_fast(
        series, parallel=False, window=4, penalty=0.1
    )
    assert np.allclose(actual, expected)


def test_distance_matrix_parallel_threshold_paths():
    rng = np.random.default_rng(32)
    small = np.ascontiguousarray(rng.normal(size=(8, 8)))
    large = np.ascontiguousarray(rng.normal(size=(18, 32)))
    assert np.allclose(
        dtw.distance_matrix_fast(small, parallel=True, use_pruning=False),
        dtw.distance_matrix_fast(small, parallel=False, use_pruning=False),
    )
    assert np.allclose(
        dtw.distance_matrix_fast(large, parallel=True, use_pruning=False),
        upstream_dtw.distance_matrix_fast(
            large, parallel=False, use_pruning=False
        ),
    )


def test_distance_matrix_ragged_compact_and_index():
    series = [
        np.array([0.0, 1, 2]),
        np.array([0.0, 1]),
        np.array([1.0, 2, 3, 4]),
        np.array([2.0, 2, 1]),
    ]
    expected = np.asarray(upstream_dtw.distance_matrix(series, compact=True))
    actual = dtw.distance_matrix(series, compact=True)
    assert np.allclose(actual, expected)
    for a in range(len(series)):
        for b in range(a + 1, len(series)):
            idx = dtw.distance_array_index(a, b, len(series))
            assert idx == upstream_dtw.distance_array_index(a, b, len(series))
            assert actual[idx] == pytest.approx(
                upstream_dtw.distance(series[a], series[b])
            )


def test_distance_matrix_block_and_upper_triangle():
    series = np.arange(30.0).reshape(5, 6)
    block = ((1, 3), (0, 5), False)
    expected = np.asarray(
        upstream_dtw.distance_matrix(series, block=block, compact=True)
    )
    actual = dtw.distance_matrix(series, block=block, compact=True)
    assert np.allclose(actual, expected)
    expected_tri = upstream_dtw.distance_matrix(series, only_triu=True)
    actual_tri = dtw.distance_matrix(series, only_triu=True)
    assert np.allclose(actual_tri, expected_tri)


def test_multivariate_distance_paths_and_matrix():
    rng = np.random.default_rng(11)
    a = rng.normal(size=(8, 3))
    b = rng.normal(size=(6, 3))
    assert dtw_ndim.distance(a, b, window=3) == pytest.approx(
        upstream_ndim.distance(a, b, window=3)
    )
    expected_d, expected_paths = upstream_ndim.warping_paths(a, b, penalty=0.2)
    actual_d, actual_paths = dtw_ndim.warping_paths(a, b, penalty=0.2)
    assert actual_d == pytest.approx(expected_d)
    assert np.allclose(actual_paths, expected_paths)
    series = rng.normal(size=(5, 7, 3))
    assert np.allclose(
        dtw_ndim.distance_matrix(series),
        upstream_ndim.distance_matrix(series),
    )


def test_warping_paths_simd_fill_tail():
    a = np.array([0.0, 1.0])
    b = np.array([0.0, 0.5])
    expected_d, expected_paths = upstream_dtw.warping_paths(a, b)
    actual_d, actual_paths = dtw.warping_paths(a, b)
    assert actual_d == pytest.approx(expected_d)
    assert np.allclose(actual_paths, expected_paths)


@pytest.mark.parametrize("a,b", [([], [1.0]), ([1.0], []), ([], [])])
def test_empty_warping_paths_remain_initialized(a, b):
    expected_d, expected_paths = upstream_dtw.warping_paths(a, b)
    actual_d, actual_paths = dtw.warping_paths(a, b)
    assert actual_d == expected_d
    assert np.allclose(actual_paths, expected_paths)


def test_dba_one_step_and_loop():
    series = np.array(
        [
            [0.0, 0, 1, 2, 1, 0],
            [0.0, 1, 2, 1, 0, 0],
            [0.0, 0, 0, 1, 2, 1],
        ]
    )
    center = series[0]
    assert np.allclose(
        dtw_barycenter.dba(series, center),
        upstream_barycenter.dba(series, center),
    )
    assert np.allclose(
        dtw_barycenter.dba_loop(series, center, max_it=5),
        upstream_barycenter.dba_loop(series, center, max_it=5),
    )


def test_dba_multivariate_and_mask():
    rng = np.random.default_rng(15)
    series = rng.normal(size=(5, 8, 2))
    mask = np.array([True, False, True, True, False])
    expected = upstream_barycenter.dba(
        series, series[0], mask=mask, use_c=False
    )
    actual = dtw_barycenter.dba(series, series[0], mask=mask)
    assert np.allclose(actual, expected)


def test_kmeans_cluster_and_means_parity():
    rng = np.random.default_rng(4)
    base = np.sin(np.linspace(0, 3, 24))
    series = np.array(
        [base + rng.normal(0, 0.03, 24) for _ in range(6)]
        + [base[::-1] + 3 + rng.normal(0, 0.03, 24) for _ in range(6)]
    )
    np.random.seed(5)
    random.seed(5)
    expected_model = UpstreamKMeans(
        2, max_it=4, max_dba_it=3, show_progress=False
    )
    expected, expected_it = expected_model.fit(series, use_parallel=False)
    np.random.seed(5)
    random.seed(5)
    actual_model = KMeans(2, max_it=4, max_dba_it=3, show_progress=False)
    actual, actual_it = actual_model.fit(series, use_parallel=False)
    assert actual == expected
    assert actual_it == expected_it
    assert all(
        np.allclose(a, b) for a, b in zip(actual_model.means, expected_model.means)
    )


def test_kmedoids_partition_parity():
    series = np.array(
        [
            [0, 0, 1, 2, 1, 0],
            [0, 1, 2, 1, 0, 0],
            [0, 0, 1, 2, 1, 0.1],
            [5, 5, 6, 7, 6, 5],
            [5, 6, 7, 6, 5, 5],
            [5, 5, 6, 7, 6, 5.1],
        ],
        dtype=float,
    )
    expected_model = UpstreamKMedoids(
        upstream_dtw.distance_matrix_fast, {}, k=2, initial_medoids=[0, 3],
        show_progress=False
    )
    actual_model = KMedoids(
        dtw.distance_matrix_fast, {}, k=2, initial_medoids=[0, 3],
        show_progress=False
    )
    expected = {frozenset(v) for v in expected_model.fit(series).values()}
    actual = {frozenset(v) for v in actual_model.fit(series).values()}
    assert actual == expected
