import numpy as np
from manim.typing import Vector3D


def unit_vector(vector: Vector3D) -> Vector3D:
    """
    Calculate the unit vector of a given 3D vector.

    Args:
        vector: A 3D vector.

    Returns:
        The unit vector of the given vector.
    """

    return vector / np.linalg.norm(vector)
