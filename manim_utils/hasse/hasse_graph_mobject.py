from typing import Hashable, List, Tuple

import networkx as nx
import numpy as np

from manim.utils.color import WHITE, ManimColor

from manim_utils.graph.movable_graph import MovableGraph
from manim_utils.hasse.hasse_graph import NXHasseGraph


class HasseGraph(MovableGraph):
    """
    Hasse graph.
    """

    def __init__(
        self,
        vertices: List[Hashable],
        edges: List[Tuple[Hashable, Hashable]],
        label_colour: ManimColor = WHITE,
        label_buff: float = 0.3,
        perform_correction: bool = True,
        iterations: int = 250,
        seed: int = 8192,
        **kwargs
    ):
        """
        Initializes a HasseGraph object.

        Args:
            vertices: List of vertices in the graph.
            edges: List of edges, where each edge is a tuple of vertices.
            label_colour: Color of the labels on the graph.
            label_buff: Buffer distance for the labels.
            perform_correction: Whether to perform layout correction.
            iterations: Number of iterations for the spring layout algorithm. Only used if
                `perform_correction` is True.
            seed: Seed value for the layout algorithm. Only used if `perform_correction` is True.
            **kwargs: Additional keyword arguments for the base Graph class.
        """

        # Generate the NetworkX hasse graph
        self._hasse_graph = NXHasseGraph(vertices, edges)

        # Update configs
        vertex_config = kwargs.pop("vertex_config", {})
        vertex_config["fill_opacity"] = 0.0

        edge_config = kwargs.pop("edge_config", {})
        edge_config["buff"] = label_buff

        layout_config = kwargs.pop("layout_config", {})
        layout_config["perform_correction"] = perform_correction
        layout_config["iterations"] = iterations
        layout_config["seed"] = seed

        # Generate the graph
        super().__init__(
            list(self._hasse_graph.nodes),
            list(self._hasse_graph.edges),
            layout=self._layout,
            labels=True,
            vertex_config=vertex_config,
            edge_config=edge_config,
            label_fill_color=label_colour,
            layout_config=layout_config,
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
