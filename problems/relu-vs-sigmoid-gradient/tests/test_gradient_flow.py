import importlib.util
from pathlib import Path

import numpy as np

_SOLUTION = Path(__file__).resolve().parents[1] / "starter" / "solution.py"
_spec = importlib.util.spec_from_file_location("solution", _SOLUTION)
assert _spec is not None and _spec.loader is not None
solution = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(solution)


def test_sigmoid_no_overflow() -> None:
    out = solution.sigmoid(np.array([-1000.0, 0.0, 1000.0]))
    assert np.all(np.isfinite(out))
    assert np.isclose(out[1], 0.5)


def test_sigmoid_grad_peaks_at_quarter() -> None:
    assert np.isclose(solution.sigmoid_grad(np.array([0.0]))[0], 0.25)
    assert np.all(solution.sigmoid_grad(np.array([-5.0, 5.0])) < 0.25)


def test_relu_grad_is_a_step_function() -> None:
    out = solution.relu_grad(np.array([-1.0, 1.0, 2.0]))
    assert np.array_equal(out, np.array([0.0, 1.0, 1.0]))


def test_sigmoid_gradient_vanishes_with_depth() -> None:
    pre_activations = np.full(6, 2.0)
    grad = solution.backprop_gradient_norm(solution.sigmoid_grad, pre_activations)
    assert abs(grad) < 0.01


def test_relu_gradient_survives_depth() -> None:
    pre_activations = np.full(6, 2.0)
    grad = solution.backprop_gradient_norm(solution.relu_grad, pre_activations)
    assert np.isclose(grad, 1.0)


def test_relu_gradient_dies_if_any_layer_is_off() -> None:
    pre_activations = np.array([2.0, 2.0, -1.0, 2.0, 2.0, 2.0])
    grad = solution.backprop_gradient_norm(solution.relu_grad, pre_activations)
    assert np.isclose(grad, 0.0)
