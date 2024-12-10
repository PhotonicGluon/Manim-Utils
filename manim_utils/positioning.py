from manim.constants import DEGREES
from manim.mobject.geometry.line import Line
from manim.mobject.mobject import Mobject


def flip_about_line(mobject: Mobject, line: Line, handle_orientation: bool = True) -> None:
    """
    Flip a mobject about a given line.

    Args:
        mobject: The mobject to be flipped.
        line: The line about which to flip the mobject.
        handle_orientation: Whether to modify the orientation and reflection of the object after
            flipping. Defaults to True.
    """

    # Shift the mobject
    center = mobject.get_center()
    foot_of_perpendicular = line.get_projection(center)
    mobject.shift(2 * (foot_of_perpendicular - center))

    # Handle orientation of the mobject
    if handle_orientation:
        mobject.flip()  # To correctly reflect any chiral stuff
        angle = line.get_angle() - 90 * DEGREES  # Our reference is the vertical, which is 90 degrees
        mobject.rotate(2 * angle)
