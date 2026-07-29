import inspect

import numpy as np
import pytest

from mojo_dtaidistance import dtw, dtw_barycenter, dtw_ndim
from mojo_dtaidistance._core import pack_series


def test_fast_signatures_cover_upstream_names():
    assert "use_pruning" in inspect.signature(dtw.distance_fast).parameters
    assert "compact" in inspect.signature(dtw.distance_matrix_fast).parameters
    assert "keep_int_repr" in inspect.signature(dtw.warping_paths_fast).parameters


def test_lists_and_float32_are_accepted():
    expected = dtw.distance(
        np.array([0, 1, 2], dtype=np.float64),
        np.array([0, 2], dtype=np.float64),
    )
    assert dtw.distance([0, 1, 2], np.array([0, 2], dtype=np.float32)) == pytest.approx(
        expected
    )


def test_lossy_or_nonreal_inputs_are_rejected():
    with pytest.raises(TypeError):
        dtw.distance(np.array([1 + 2j]), np.array([1 + 2j]))
    with pytest.raises(ValueError):
        dtw.distance(np.array([2**53 + 1], dtype=np.int64), [0])
    with pytest.raises(ValueError):
        dtw.distance(
            np.array([np.longdouble("1.0000000000000000001")]),
            np.array([np.longdouble("1.0")]),
        )


def test_contiguous_batch_packing_is_zero_copy():
    series = np.arange(60.0).reshape(5, 12)
    data, offsets, arrays, ndim = pack_series(series)
    assert np.shares_memory(data, series)
    assert offsets.tolist() == [0, 12, 24, 36, 48, 60]
    assert len(arrays) == 5
    assert ndim == 1


def test_invalid_shapes_and_inner_distance():
    with pytest.raises(ValueError):
        dtw.distance(np.zeros((2, 2)), np.zeros((2, 2)))
    with pytest.raises(ValueError):
        dtw.distance([0, 1], [0, 1], inner_dist="manhattan")
    with pytest.raises(NotImplementedError):
        dtw.warping_paths_fast([0, 1], [0, 1], compact=True)
    with pytest.raises(NotImplementedError):
        dtw.warping_path_fast([0, 1], [0, 1], use_lowmem=True)
    with pytest.raises(TypeError):
        dtw.distance([0, 1], [0, 1], window=1.5)
    with pytest.raises(ValueError):
        dtw.distance([0, 1], [0, 1], penalty=-1)


def test_settings_roundtrip():
    settings = dtw.DTWSettings(window=3, penalty=0.5, psi=(1, 0, 2, 0))
    assert settings.split_psi() == (1, 0, 2, 0)
    assert dtw.DTWSettings.wrap({"dtw_settings": settings}) is settings
    assert settings.c_kwargs()["window"] == 3


def test_documented_helper_surface():
    a = np.array([0.0, 1, 2, 1])
    b = np.array([0.0, 1, 1, 2])
    path, distance = dtw.warping_path_fast(a, b, include_distance=True)
    _, paths = dtw.warping_paths(a, b)
    assert path == dtw.best_path(paths)
    assert dtw.best_path2(paths) == path
    assert dtw.warping_amount(path) >= 0
    warped, returned_path = dtw.warp(a, b, path=path)
    assert returned_path == path
    assert len(warped) == len(b)
    penalized = dtw.warping_path_penalty(a, b, penalty_post=0.5)
    assert penalized[0] >= distance
    assert penalized[1] == path

    compact = dtw.distance_matrix([a, b], compact=True)
    matrix = dtw.distances_array_to_matrix(compact, 2)
    assert matrix[0, 1] == pytest.approx(distance)
    assert dtw.distance_array_index(0, 1, 2) == 0
    assert dtw.lb_keogh(a, b) <= distance
    assert dtw.ub_euclidean(a, b) >= 0

    multivariate = np.column_stack((a, a + 1))
    assert dtw_ndim.distance_fast(multivariate, multivariate) == 0
    ndim_path = dtw_ndim.warping_path_fast(multivariate, multivariate)
    assert ndim_path == [(i, i) for i in range(len(a))]
    assert dtw_ndim.ub_euclidean(multivariate, multivariate) == 0

    averages, history = dtw_barycenter.dba_loop(
        [a, b], max_it=2, keep_averages=True
    )
    assert averages.shape == a.shape
    assert history
