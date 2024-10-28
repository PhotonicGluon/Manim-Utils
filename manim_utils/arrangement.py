from manim import RIGHT, TAU, UP, Mobject, np


def arrange_circular(*mobjects: Mobject, clockwise: bool = False, radius: float = 1.0, offset: float = 0.0):
    """
    Arranges the mobjects in a circle.

    The first mobject will be placed at the **top** of the circle.

    Args:
        mobjects: Mobjects to arrange.
        clockwise: Whether the arrangement should be clockwise instead of anticlockwise.
        radius: The radius of the circle to arrange the objects around.
        offset: The offset of the arrangement.
    """

    num_mobjects = len(mobjects)
    multiplier = -1 if clockwise else 1
    offset += TAU / 4

    for i, mobject in enumerate(mobjects):
        theta = TAU * multiplier * i / num_mobjects + offset
        right = np.cos(theta) * radius
        up = np.sin(theta) * radius
        mobject.shift(right * RIGHT + up * UP)
