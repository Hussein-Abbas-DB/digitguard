import numpy as np
import pytest
from sklearn.datasets import load_digits

from digitguard.core import calibrated, conformal_cutoff, infer, prediction_sets, select_threshold
from digitguard.train import split_indices, stress_cases


def test_splits_partition_all_rows_and_repeat():
    _, labels = load_digits(return_X_y=True)
    first = split_indices(labels)
    second = split_indices(labels)
    assert np.array_equal(np.sort(np.concatenate(list(first.values()))), np.arange(len(labels)))
    for name in first:
        assert np.array_equal(first[name], second[name])
        assert set(labels[first[name]]) == set(range(10))


def test_temperature_keeps_order_and_normalizes():
    values = np.array([[0.8, 0.2], [0.4, 0.6]])
    assert np.allclose(calibrated(values, 1), values)
    sharpened = calibrated(values, 0.5)
    assert np.allclose(sharpened.sum(axis=1), 1)
    assert (sharpened.max(axis=1) > values.max(axis=1)).all()


def test_finite_sample_quantile_and_inclusive_sets():
    probabilities = np.array([[0.75, 0.25], [0.5, 0.5], [0.25, 0.75]])
    labels = np.array([0, 0, 0])
    cutoff = conformal_cutoff(probabilities, labels, coverage=0.5)
    assert cutoff == 0.5
    assert prediction_sets(probabilities, cutoff).tolist() == [[True, False], [True, True], [False, True]]
    assert conformal_cutoff(probabilities, labels) == 1


def test_threshold_excludes_wrong_low_confidence_rows():
    p = np.vstack([np.tile([0.7, 0.3], (10, 1)), np.tile([0.95, 0.05], (35, 1))])
    y = np.array([1] * 10 + [0] * 35)
    assert select_threshold(p, y, cutoff=0.4) == 0.95
    assert select_threshold(p, 1 - y, cutoff=0.4) > 1


def test_blank_input_skips_inference():
    assert infer({}, [0] * 64)["reason"] == "blank_input"


@pytest.mark.parametrize("pixels", [[0] * 63, [np.nan] * 64, [17] * 64])
def test_invalid_pixels(pixels):
    with pytest.raises(ValueError):
        infer({}, pixels)


def test_stress_cases_preserve_dimensions_and_do_not_mutate_input():
    values = np.full((2, 64), 8.0)
    cases = stress_cases(values)
    assert (values == 8).all()
    assert all(x.shape == values.shape and x.min() >= 0 and x.max() <= 16 for x in cases.values())
    assert (cases["shift_right_1px"].reshape(-1, 8, 8)[:, :, 0] == 0).all()
    assert (cases["center_occlusion_2x2"].reshape(-1, 8, 8)[:, 3:5, 3:5] == 0).all()
