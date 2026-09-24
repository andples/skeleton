import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    raise NotImplementedError


def sigmoid_grad(x: np.ndarray) -> np.ndarray:
    raise NotImplementedError


def relu_grad(x: np.ndarray) -> np.ndarray:
    raise NotImplementedError


def backprop_gradient_norm(grad_fn, pre_activations: np.ndarray, upstream_grad: float = 1.0) -> float:
    raise NotImplementedError
