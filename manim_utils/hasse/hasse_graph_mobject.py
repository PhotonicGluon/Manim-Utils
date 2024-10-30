from typing import Hashable, List, Tuple

import networkx as nx
import numpy as np
from manim.mobject.graph import Graph
from manim.utils.color import WHITE, ManimColor

from manim_utils.hasse.hasse_graph import NXHasseGraph


class HasseGraph(Graph):
    """
    Hasse graph.
    """

    def __init__(
        self,
        vertices: List[Hashable],
        edges: List[Tuple[Hashable, Hashable]],
        label_colour: ManimColor = WHITE,
        label_buff: float = 0.3,
        **kwargs
    ):
        # Generate the NetworkX hasse graph
        self._hasse_graph = NXHasseGraph(vertices, edges)

        # Generate the graph
        super().__init__(
            list(self._hasse_graph.nodes),
            list(self._hasse_graph.edges),
            layout=self._layout,
            labels=True,
            vertex_config={"fill_opacity": 0.0},
            edge_config={"buff": label_buff},
            label_fill_color=label_colour,
            **kwargs
        )

    # Helper methods
    def _layout(self, graph: nx.Graph, scale: float = 2.0, scale_x: float = 2.0, scale_y: float = 0.75, **kwargs):
        # Generate the base layout
        layout_dict = self._hasse_graph._hasse_layout(**kwargs)

        # Convert all positions to 3D, and scale appropriately
        for node, pos in layout_dict.items():
            layout_dict[node] = scale * np.array([scale_x * pos[0], scale_y * pos[1], 0.0])

        return layout_dict
