from typing import Hashable

from manim.mobject.graph import Graph
from manim.typing import Vector3D


class MovableGraph(Graph):
    def shift_vertex(self, vertex: Hashable, *vectors: Vector3D):
        """
        Shifts the given vertex, along with all its connected edges.

        Args:
            vertex: The vertex to shift.
            *vectors: The vectors to shift the vertex by.
        """

        # First shift the vertex itself
        self[vertex].shift(*vectors)

        # Find all edges that are connected to this vertex
        in_edges = set(line for edge, line in self.edges.items() if edge[0] == vertex)
        out_edges = set(line for edge, line in self.edges.items() if edge[1] == vertex)
        edges = in_edges.union(out_edges)

        # Shift all edges that are connected to this vertex
        for edge in edges:
            start, end = edge.get_start_and_end()

            if edge in in_edges:
                for vector in vectors:
                    start += vector

            if edge in out_edges:
                for vector in vectors:
                    end += vector

            edge.put_start_and_end_on(start, end)
