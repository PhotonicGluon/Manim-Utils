import warnings
from collections import defaultdict
from functools import cached_property
from typing import Dict, Hashable, List, Tuple

import networkx as nx
import numpy as np
from ordered_set import OrderedSet


class NXDirectedAcyclicGraph(nx.DiGraph):
    """
    Directed Acyclic Graph (DAG) class.
    """

    def __init__(self, nodes: List[Hashable], edges: List[Tuple[Hashable, Hashable]], **attr):
        """
        Initializes a new directed acyclic graph (DAG).

        Args:
            nodes: list of nodes.
            edges: list of edges.

        Raises:
            NetworkXError: if the graph is not a directed acyclic graph (DAG).
        """

        # Add the nodes and edges
        super().__init__(**attr)
        self.add_nodes_from(nodes)
        self.add_edges_from(edges)

        # Ensure that we indeed have a DAG
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


class NXHasseGraph(NXDirectedAcyclicGraph):
    def __init__(self, nodes: List[Hashable], edges: List[Tuple[Hashable, Hashable]], **attr):
        """
        Initializes a new Hasse graph.

        Args:
            nodes: list of nodes.
            edges: list of edges.

        Raises:
            NetworkXError: if the graph is not a directed acyclic graph (DAG).
        """

        # First initialize it as a DAG
        super().__init__(nodes, edges, **attr)

        # Remove any edges that are implied by transitivity
        self._trim_transitive_edges()

    # Helper methods
    def _trim_transitive_edges(self):
        """
        Trim any edges that are implied by transitivity in the graph.

        This ensures that the graph is a Hasse diagram, which is a directed acyclic graph (DAG)
        that has no "redundant" edges: if there is a path from u to v, then there is no edge directly
        from u to v.

        The algorithm works by first computing the transitive closure of the graph, and then
        removing any edge (u, v) if there is still a path from u to v in the transitive closure.
        """

        # Get the transitive closure of the graph
        transitive_closure: nx.DiGraph = nx.transitive_closure(nx.DiGraph(self))

        # Remove any edge (u, v) that still allows a path from u to v within the transitive closure
        edges = list(self.edges)  # This will change!
        for u, v in edges:
            # Check if path still exists
            transitive_closure.remove_edge(u, v)
            if nx.has_path(transitive_closure, u, v):
                warning = f"Provided Hasse diagram has transitive edge ({u}, {v}). Removing."
                warnings.warn(warning)
                self.remove_edge(u, v)
            else:
                transitive_closure.add_edge(u, v)

    def _get_best_generation_ordering(
        self, pos: np.ndarray, iterations: int = 250, seed: int = 8192
    ) -> Dict[int, List[Hashable]]:
        """
        Given a set of positions, use spring layout to find the best possible generation ordering
        for the graph.

        Args:
            pos: The positions of the nodes in the graph.
            iterations: The number of iterations to use in the spring layout algorithm.
            seed: The seed value to use in the spring layout algorithm.

        Returns:
            The best possible generation ordering for the graph, as a dictionary mapping generation
            number to a list of nodes in that generation.
        """

        # First convert the DAG into a regular graph
        graph = nx.Graph(self)

        # Use spring layout (i.e., Fruchterman-Reingold) to get a good node positioning
        positions = nx.spring_layout(graph, pos=dict(zip(self, pos)), iterations=iterations, seed=seed)

        # Arrange the given positions in a list
        positions = [(node, pos) for node, pos in positions.items()]

        # Sort by x coordinate
        positions.sort(key=lambda t: t[1][0])

        # Now we can order each generation properly
        best_ordering = defaultdict(list)
        for node, _ in positions:
            best_ordering[self.node_generations[node]].append(node)

        return dict(best_ordering)

    def _hasse_layout(
        self,
        perform_correction: bool = True,
        iterations: int = 250,
        seed: int = 8192,
        horizontal_spacing: float = 0.5,
        vertical_spacing: float = 1,
        center: bool = True,
    ) -> Dict[Hashable, np.ndarray]:
        """
        Computes the best possible layout of the graph in 2D space by first running spring layout
        on the graph, then ordering the nodes in each generation such that the crossings are
        minimized.

        Args:
            perform_correction: Whether to perform the correction step.
            iterations: The number of iterations to use in the spring layout algorithm.
            seed: The seed value to use in the spring layout algorithm.
            horizontal_spacing: The amount of horizontal space between each node.
            vertical_spacing: The amount of vertical space between each generation.
            center: Whether the final layout should be centred at the origin.

        Returns:
            A dictionary mapping each node to its final position in 2D space.
        """

        # First get data from the computed DAG
        num_generations = len(self.generations)
        num_nodes = len(self)

        if perform_correction:
            # Determine the nodes' initial positions
            pos = np.empty((num_nodes, 2))
            for i, node in enumerate(self):
                gen = self.node_generations[node]
                node_index = self.generations[gen].index(node)

                # Determine the node's initial position
                x = horizontal_spacing * ((node_index + 1) - self.generation_sizes[gen] / 2)
                y = vertical_spacing * ((gen + 1) - num_generations / 2)
                pos[i] = x, y

            # Now we can get the best ordering of the nodes in each generation
            best_ordering = self._get_best_generation_ordering(pos, iterations=iterations, seed=seed)
        else:
            best_ordering = self.generations

        # Now position the nodes in the best order
        pos = np.empty((num_nodes, 2))
        for i, node in enumerate(self):
            gen = self.node_generations[node]
            node_index = best_ordering[gen].index(node)

            # Determine the node's final position
            x = horizontal_spacing * ((node_index + 1) - self.generation_sizes[gen] / 2)
            y = vertical_spacing * ((gen + 1) - num_generations / 2)
            pos[i] = x, y

        # Adjust to be centred
        if center:
            # Get the range of x and y
            x_max, y_max = np.max(pos, axis=0)
            x_min, y_min = np.min(pos, axis=0)

            # Center all the nodes
            pos -= np.array([(x_max + x_min) / 2, (y_max + y_min) / 2])

        # Generate final layout dictionary
        return dict(zip(self, pos))
