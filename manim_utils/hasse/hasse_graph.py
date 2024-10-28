import random
from typing import Any, Hashable, List, Tuple

import networkx as nx
import numpy as np
from manim.mobject.graph import Graph
from manim.typing import Point3D
from manim.utils.color import WHITE, ManimColor

from manim_utils.hasse.augmented_dag import DAG

# TODO: Remove
random.seed(42)


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
        **kwargs
    ):
        # Generate the internal DAG data
        self._dag = DAG(vertices, edges)

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
    def _hasse_layout(
        self, graph: nx.Graph, level_buff: float = 2, *args: Any, **kwargs: Any
    ) -> dict[Hashable, Point3D]:
        # TODO: Update layout

        dag_height = self._dag.height

        layout_dict = {}
        for i, node in enumerate(graph):
            layout_dict[node] = np.array(
                [i - len(graph) / 2 + 2 * random.random(), level_buff * (self._dag.get_level(node) - dag_height / 2), 0]
            )
        print(layout_dict)
        return layout_dict
