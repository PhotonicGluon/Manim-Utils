from typing import Hashable, Union

import numpy as np
from manim.constants import ORIGIN
from manim.mobject.graph import Graph
from manim.mobject.mobject import Mobject
from manim.typing import Vector3D, Point3D


class MovableGraph(Graph):
    def move_vertex(
        self,
        vertex: Hashable,
        point_or_mobject: Union[Point3D, Mobject],
        aligned_edge: Vector3D = ORIGIN,
        coor_mask: Vector3D = np.array([1, 1, 1]),
    ):
        """
        Move center of the vertex to certain Point3D.

        Args:
            vertex: The vertex to move.
            point_or_mobject: A point or a mobject that the vertex should be moved to.
            aligned_edge: The edge of the vertex that should be aligned with the target position.
                Defaults to ORIGIN.
            coor_mask: A 3D vector that determines which coordinates should be changed. Defaults to
                [1, 1, 1], which means all coordinates are changed.
        """

        # Determine the target position
        if isinstance(point_or_mobject, Mobject):
            target = point_or_mobject.get_critical_point(aligned_edge)
        else:
            target = point_or_mobject

        # Determine alignment
        point_to_align = self[vertex].get_critical_point(aligned_edge)

        # Then shift the vertex
        self.shift_vertex(vertex, ((target - point_to_align) * coor_mask))

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
