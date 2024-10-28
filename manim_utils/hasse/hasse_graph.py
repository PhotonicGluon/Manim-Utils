from typing import Hashable, List, Tuple

from manim.constants import DOWN, UP
from manim.mobject.graph import DiGraph

from manim_utils.hasse.augmented_dag import DAG


class HasseGraph(DiGraph):
    """
    Hasse graph.
    """

    def __init__(
        self, vertices: List[Hashable], edges: List[Tuple[Hashable, Hashable]], level_buff: float = 0.5, **kwargs
    ):
        super().__init__(vertices, edges, **kwargs)

        # Generate the internal DAG data
        self._dag = DAG(vertices, edges)

        # Other attributes
        self._level_buff = level_buff

        # Preprocess
        # self._arrange_vertices()

    # Helper methods
    def _arrange_vertices(self):
        # Compute level 0 offset from the center
        level_0_offset = self._dag.height / 2 * self._level_buff * DOWN

        for vertex, mobject in self.vertices.items():
            mobject.shift(level_0_offset + self._dag.get_level(vertex) * self._level_buff * UP)
