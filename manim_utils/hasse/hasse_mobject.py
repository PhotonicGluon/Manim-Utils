from collections import defaultdict
from typing import Hashable, List, Literal, Tuple

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
        centering: bool = True,
        seed: int = 10,
        **kwargs
    ):
        # Generate the internal DAG data
        self._dag = DAG(vertices, edges)

        # Set internal attributes
        self._intra_level_spacing = intra_level_spacing
        self._inter_level_spacing = inter_level_spacing
        self._centering = centering
        self._seed = seed

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
        # TODO: Add docs

        # First get data from the computed DAG
        dag_height = self._dag.height
        dag_level_sizes = self._dag.level_sizes
        num_nodes = self._dag.num_vertices

        # Then start laying out the nodes properly
        levels_node_counts = defaultdict(lambda: 1)
        pos = np.empty((num_nodes, 2))
        rand = np.random.RandomState(self._seed)
        for i, node in enumerate(graph):
            # Get the node's level
            node_level = self._dag.get_level(node)

            # Determine the node's position
            x = self._intra_level_spacing * (levels_node_counts[node_level] - dag_level_sizes[node_level] / 2)
            x += 0.9 * rand.rand() + 0.1  # Add randomness in the uniform interval [0.1, 1.0)
            # TODO: Enforce that all nodes that came before indeed come before

            y = self._inter_level_spacing * (node_level - dag_height / 2)

            # Note it down in the layout dictionary
            pos[i] = x, y

            # Update
            levels_node_counts[node_level] += 1

        # Use a modified Fruchterman-Reingold force-directed algorithm to nicely arrange nodes along each level
        pos = self._fruchterman_reingold(nx.to_numpy_array(graph, weight="weight"), pos)

        # Adjust to be centred
        if self._centering:
            # Get the range of x and y
            x_max, y_max = np.max(pos, axis=0)
            x_min, y_min = np.min(pos, axis=0)

            # Center all the nodes
            pos -= np.array([(x_max + x_min) / 2, (y_max + y_min) / 2])

        # Generate final layout dictionary
        layout_dict = {}
        for i, node in enumerate(graph):
            layout_dict[node] = np.array([*pos[i], 0])

        # print(layout_dict)
        return layout_dict

    @staticmethod
    def _fruchterman_reingold(
        A: np.ndarray,
        pos: np.ndarray,
        min_dist: float = 1.5,
        temperature: float = 0.05,
        freeze: Literal["x", "y"] = "y",
        iterations: int = 50,
    ):
        # TODO: Add docs
        # See https://networkx.org/documentation/stable/_modules/networkx/drawing/layout.html#spring_layout

        # Get the number of nodes from our adjacency matrix
        num_nodes, _ = A.shape

        # Make sure positions are of same type as matrix
        pos = pos.astype(A.dtype)

        # Optimal distance between nodes
        optimal_dist = np.sqrt(1.0 / num_nodes)

        # Calculate initial temperature based on domain size
        t = max(max(pos.T[0]) - min(pos.T[0]), max(pos.T[1]) - min(pos.T[1])) * temperature

        # Simple cooling scheme: linearly step down by `dt` on each iteration so last iteration is size `dt`
        dt = t / (iterations + 1)

        # Perform Fruchterman-Reingold (fast)
        delta = np.zeros((pos.shape[0], pos.shape[0], pos.shape[1]), dtype=A.dtype)
        for i in range(iterations):
            # Matrix of difference between points
            delta = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]

            # Distance between points
            distance = np.linalg.norm(delta, axis=-1)
            np.clip(distance, min_dist, None, out=distance)  # Enforce minimum distance

            # Displacement "force"
            displacement = np.einsum(
                "ijk,ij->ik", delta, (optimal_dist * optimal_dist / distance**2 - A * distance / optimal_dist)
            )

            # Calculate initial delta for positions
            length = np.linalg.norm(displacement, axis=-1)
            length = np.where(length < min_dist, temperature, length)
            delta_pos = np.einsum("ij,i->ij", displacement, t / length)

            # If freezing specified, freeze x or y
            if freeze == "x":
                freeze_index = 0
            else:
                freeze_index = 1
            delta_pos[:, freeze_index] = 0

            # Adjust positions
            pos += delta_pos

            # Cool temperature
            t -= dt
        return pos
