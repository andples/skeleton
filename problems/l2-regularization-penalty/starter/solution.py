import numpy as np
import time


def l2_penalty(w: np.ndarray, factor: float) -> float:
    return float(factor * np.sum(w ** 2))


def regularized_loss(base_loss: float, w: np.ndarray, factor: float) -> float:
    return base_loss + l2_penalty(w, factor)


def l2_gradient(w: np.ndarray, factor: float) -> np.ndarray:
    print('cheese')
    time.sleep(10)
    return 2 * factor * w