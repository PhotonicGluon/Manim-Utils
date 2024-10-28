import random
from typing import Any, Hashable, List, Tuple, Union

import networkx as nx
import numpy as np
from manim.mobject.graph import DiGraph
from manim.typing import Point3D

from manim_utils.hasse.augmented_dag import DAG


class HasseGraph(DiGraph):
    """
    Hasse graph.
    """

    def __init__(self, vertices: List[Hashable], edges: List[Tuple[Hashable, Hashable]], **kwargs):
        # Generate the internal DAG data
        self._dag = DAG(vertices, edges)

        # Generate the Digraph
        super().__init__(vertices, edges, layout=self._hasse_layout, **kwargs)

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
