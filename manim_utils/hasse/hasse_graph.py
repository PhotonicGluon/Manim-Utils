from collections import defaultdict
from typing import Hashable, List, Tuple

import networkx as nx
import numpy as np
from manim.mobject.graph import Graph
from manim.typing import Point3D
from manim.utils.color import WHITE, ManimColor

from manim_utils.hasse.augmented_dag import DAG


class HasseGraph(Graph):
    """
    Hasse graph.
    """

    def __init__(
        self,
        vertices: List[Hashable],
        edges: List[Tuple[Hashable, Hashable]],
        label_colour: ManimColor = WHITE,
        label_buff: float = 0.25,
        intra_level_spacing: float = 1.0,
        inter_level_spacing: float = 1.0,
        **kwargs
    ):
        # Generate the internal DAG data
        self._dag = DAG(vertices, edges)

        # Set internal attributes
        self._intra_level_spacing = intra_level_spacing
        self._inter_level_spacing = inter_level_spacing

        # Generate the graph
        super().__init__(
            vertices,
            edges,
            layout=self._hasse_layout,
            labels=True,
            vertex_config={"fill_opacity": 0.0},
            edge_config={"buff": label_buff},
            label_fill_color=label_colour,
            **kwargs
        )

    # Helper methods
    def _hasse_layout(self, graph: nx.Graph, *args, **kwargs) -> dict[Hashable, Point3D]:
        # TODO: Update layout

        # First get data from the computed DAG
        dag_height = self._dag.height
        dag_level_sizes = self._dag.level_sizes

        # Then start laying out the nodes properly
        levels_node_counts = defaultdict(lambda: 1)
        layout_dict = {}
        for i, node in enumerate(graph):
            # Get the node's level
            node_level = self._dag.get_level(node)

            # Determine the node's position
            x = self._intra_level_spacing * (levels_node_counts[node_level] - dag_level_sizes[node_level] / 2)
            y = self._inter_level_spacing * (node_level - dag_height / 2)

            # Note it down in the layout dictionary
            layout_dict[node] = np.array([x, y, 0])

            # Update
            levels_node_counts[node_level] += 1

        print(layout_dict)  # TODO: Remove
        return layout_dict
