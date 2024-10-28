from collections import defaultdict
from typing import Any, Dict, List, Set, Tuple


class Node:
    """
    Graph node.
    """

    def __init__(self, obj: Any):
        self.obj = obj
        self.level = -1

    # Magic methods
    def __repr__(self):
        return f"Node({self.obj}, {self.level})"

    def __str__(self):
        return f"Node({self.obj})"

    def __hash__(self):
        return hash(self.obj)


class DAG:
    """
    Augmented Directed Acyclic Graph class.
    """

    def __init__(self, vertices: List[Any], edges: List[Tuple[Any, Any]]):
        # Convert the inputs to node representation
        vertex_to_node = {vertex: Node(vertex) for vertex in vertices}
        vertices = [vertex_to_node[v] for v in vertices]
        edges = [(vertex_to_node[a], vertex_to_node[b]) for (a, b) in edges]

        # Define attributes
        self._vertices: List[Node] = vertices
        self._edges: List[Tuple[Node, Node]] = edges
        self._adj_list: Dict[Node, List[Node]] = self._generate_adjacency_list()

        # Preprocessing
        self._compute_node_levels()

    # Properties
    @property
    def vertices(self) -> List[Any]:
        return [node.obj for node in self._vertices]

    @property
    def num_vertices(self) -> int:
        return len(self._vertices)

    @property
    def edges(self) -> List[Tuple[Any, Any]]:
        return [(a.obj, b.obj) for (a, b) in self._edges]

    @property
    def num_edges(self) -> int:
        return len(self._edges)

    # Helper methods
    def _generate_adjacency_list(self) -> Dict[Node, List[Node]]:
        """
        Generates the adjacency list from the list of edges.
        """

        adj_list = {node.obj: [] for node in self._vertices}
        for edge in self._edges:
            adj_list[edge[0].obj].append(edge[1])
        return dict(adj_list)

    def _find_sources(self) -> Set[Node]:
        """
        Finds all the nodes that are sources (i.e., has in-degree of 0).

        :return: set of node indices
        """

        sources = set(self._vertices.copy())
        for _, dest in self._edges:
            sources.discard(dest)

        return sources

    def _compute_node_levels(self):
        """
        Computes the levels of all the nodes in a DAG.

        :raises ValueError: if there is a cycle in the graph
        """

        # First find all the source nodes
        sources = self._find_sources()

        # Push all the source nodes onto a stack
        stack = [(source, -1) for source in sources]  # First is node, second is parent node's level

        # Process nodes on the stack
        """
        A note why `max_iter` is `1/2 * V * (V-1)`.
        
        If we indeed have a DAG,
        - the first node can connect to the (V-1) other nodes;
        - the second node can connect to the (V-2) other nodes (can't connect to previous node as
          otherwise we have a cycle);
        - the third node can connect to the (V-3) other nodes (can't connect to previous nodes as
          otherwise we have a cycle);
        - etc.
        Thus the maximum number of iterations for a true DAG with DFS is (V-1) + (V-2) + ... + 1
        which is `1/2 * V * (V-1)`.
        """
        max_iter = self.num_vertices * (self.num_vertices - 1) // 2
        iter_count = 0
        while stack and iter_count < max_iter:
            print(stack)
            # Get the next node to process
            node, parent_level = stack.pop()

            # Compute new level for the node
            new_level = max(node.level, parent_level + 1)
            node.level = new_level

            # Add node's children to the stack
            for node in self._adj_list[node.obj]:
                stack.append((node, new_level))

            iter_count += 1

        # There is a cycle if there are still things to process after `max_iter` iterations, or if some nodes are
        # missing level information
        if stack:
            raise ValueError("There is a cycle in the provided DAG")
        for node in self._vertices:
            if node.level == -1:
                raise ValueError("There is a cycle in the provided DAG")


if __name__ == "__main__":
    # Define the nodes
    vertices = [1, 2, 3, 4, 5, 6]
    edges = [
        (1, 4),
        (2, 3),
        (3, 5),
        (4, 6),
        (5, 6),
        # (6, 3),  # WARN: (6,3) makes a cycle
        # (5, 2),  # WARN: (5,2) makes cycle and removes a source
    ]
    dag = DAG(vertices, edges)
    print(dag._vertices)
