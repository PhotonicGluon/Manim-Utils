from collections import defaultdict
from functools import cached_property
from typing import Dict, Hashable, Literal

import networkx as nx
import numpy as np
from ordered_set import OrderedSet


class DirectedAcyclicGraph(nx.DiGraph):
    """
    Directed Acyclic Graph (DAG) class.
    """

    def __init__(self, incoming_graph_data=None, **attr):
        """
        Initializes a new directed acyclic graph (DAG).

        Args:
            incoming_graph_data: input graph
            **attr: additional attributes

        Raises:
            NetworkXError: if the graph is not a DAG.
        """

        # First ensure that we indeed have a DAG
        super().__init__(incoming_graph_data, **attr)

        if not nx.is_directed_acyclic_graph(self):
            raise nx.NetworkXError("Graph is not a directed acyclic graph (DAG)")

    # Properties
    @cached_property
    def generations(self) -> Dict[int, OrderedSet[Hashable]]:
        """
        Computes the generations of nodes in the directed acyclic graph (DAG).

        Returns:
            A dictionary mapping each generation to an OrderedSet of nodes that belong to that
            generation, based on topological sorting.
        """

        gens = defaultdict(OrderedSet)
        for gen, nodes in enumerate(nx.topological_generations(self)):
            gens[gen].update(nodes)

        return dict(gens)

    @cached_property
    def generation_sizes(self) -> Dict[int, int]:
        """
        Computes the size of each generation in the directed acyclic graph (DAG).

        Returns:
            A dictionary mapping each generation to the number of nodes that belong to that
            generation, based on topological sorting.
        """

        return {gen: len(nodes) for gen, nodes in self.generations.items()}

    @cached_property
    def node_generations(self) -> Dict[Hashable, int]:
        """
        Computes a mapping of each node to its respective generation in the directed acyclic graph
        (DAG).

        Returns:
            A dictionary mapping each node to the generation number it belongs to, based on
            topological sorting.
        """

        node_gens = {}
        for gen, nodes in self.generations.items():
            for node in nodes:
                node_gens[node] = gen
        return node_gens


class HasseGraph(DirectedAcyclicGraph):
    def __init__(self, incoming_graph_data=None, **attr):
        super().__init__(incoming_graph_data, **attr)

    # Helper methods
    def _hasse_layout(
        self, intra_level_spacing: float = 0.25, inter_level_spacing: float = 1.0, center: bool = True
    ) -> dict[Hashable, np.ndarray]:
        # TODO: Add docs

        # First get data from the computed DAG
        num_generations = len(self.generations)
        num_nodes = len(self)

        # Then start laying out the nodes properly
        pos = np.empty((num_nodes, 2))
        # rand = np.random.RandomState(8192)  # TODO: Change
        for i, node in enumerate(self):
            gen = self.node_generations[node]
            node_index = self.generations[gen].index(node)

            # Determine the node's position
            x = intra_level_spacing * ((node_index + 1) - self.generation_sizes[gen] / 2)
            # x += 0.9 * rand.rand() + 0.1  # Add randomness in the uniform interval [0.1, 1.0)
            # TODO: Enforce that all nodes that came before indeed come before

            y = inter_level_spacing * ((gen + 1) - num_generations / 2)

            # Note it down in the layout dictionary
            pos[i] = x, y

        # Use a modified Fruchterman-Reingold force-directed algorithm to nicely arrange nodes along each level
        pos = self._fruchterman_reingold(nx.to_numpy_array(self, weight="weight"), pos)

        # Adjust to be centred
        if center:
            # Get the range of x and y
            x_max, y_max = np.max(pos, axis=0)
            x_min, y_min = np.min(pos, axis=0)

            # Center all the nodes
            pos -= np.array([(x_max + x_min) / 2, (y_max + y_min) / 2])

        # Generate final layout dictionary
        return dict(zip(self, pos))

    @staticmethod
    def _fruchterman_reingold(
        A: np.ndarray,
        pos: np.ndarray,
        min_dist: float = 0.01,
        temperature: float = 0.1,
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
