import numpy as np
import sys

# ==============================================================================
# Level 1: Reshaping, Transposing & Dimension Manipulation
# ==============================================================================

def drill_01_flatten(x: np.array) -> np.array:
    """
    Given an input tensor `x` of arbitrary shape (e.g. (3, 4)), flatten it
    into a 1D tensor using dynamic shape inference (-1).

    Target shape: (total_elements,)
    """
    return x.reshape(-1)
