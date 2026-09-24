import numpy as np
from pathlib import Path
import importlib

_SOLUTION = Path(__file__).resolve().parents[1] / "starter" / "solution.py"
_spec = importlib.util.spec_from_file_location("solution", _SOLUTION)
assert _spec is not None and _spec.loader is not None
solution = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(solution)

def _check_not_none(out, name):
    if out is None:
        raise NotImplementedError(f"{name} returned None. Implement the function.")


def test_drill_01():
    x = np.arange(12).reshape(3, 4)
    res = solution.drill_01_flatten(x)
    _check_not_none(res, "drill_01_flatten")
    assert res.shape == (12,), f"Expected shape (12,), got {res.shape}"
    assert np.all(res == np.arange(12)), "Values do not match expected order"
