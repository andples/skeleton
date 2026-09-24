import importlib.util
from pathlib import Path

import numpy as np

_SOLUTION = Path(__file__).resolve().parents[1] / "starter" / "solution.py"
_spec = importlib.util.spec_from_file_location("solution", _SOLUTION)
assert _spec is not None and _spec.loader is not None
solution = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(solution)


def test_zero_factor_gives_zero_penalty() -> None:
    w = np.array([1.0, -2.0, 3.0])
    assert solution.l2_penalty(w, 0.0) == 0.0


def test_penalty_scales_with_square_of_weights() -> None:
    w = np.array([1.0, 2.0])
    small = solution.l2_penalty(w, 0.01)
    large = solution.l2_penalty(w * 2, 0.01)
    assert np.isclose(large, 4.0 * small)


def test_penalty_matches_formula() -> None:
    w = np.array([[1.0, 2.0], [3.0, 4.0]])
    factor = 0.1
    expected = factor * float(np.sum(w**2))
    assert np.isclose(solution.l2_penalty(w, factor), expected)


def test_regularized_loss_adds_penalty_on_top() -> None:
    w = np.array([1.0, 1.0])
    factor = 0.5
    base_loss = 2.0
    expected = base_loss + solution.l2_penalty(w, factor)
    assert np.isclose(solution.regularized_loss(base_loss, w, factor), expected)


def test_gradient_shape_and_zero_at_zero() -> None:
    w = np.zeros((3, 3))
    grad = solution.l2_gradient(w, 0.01)
    assert grad.shape == w.shape
    assert np.allclose(grad, 0.0)


def test_gradient_matches_weight_decay_formula() -> None:
    w = np.array([2.0, -3.0])
    factor = 0.1
    expected = 2 * factor * w
    assert np.allclose(solution.l2_gradient(w, factor), expected)
