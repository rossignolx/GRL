import networkx as nx
import numpy as np
import itertools

from typing import Tuple, List


def connected_to_triangle_not_neighbour(
        graph: nx.Graph) -> Tuple[List[int], int]:
    node_triangles = list(nx.triangles(graph).values())
    triangle_present = sum(node_triangles) > 0
    if not triangle_present:
        zeros = [0 for _ in range(len(graph))]
        return zeros, 0

    present = False
    res = []
    for node in graph:
        neighbors = graph.neighbors(node)
        neighbor_triangle = any(node_triangles[x] > 0 for x in neighbors)
        label = not neighbor_triangle

        present = label or present

        res.append(int(label))

    return res, int(present)


def label_graph(
        graph: nx.Graph,
        node_labels: List[int],
        graph_labels: int) -> nx.Graph:

    nx.set_node_attributes(graph, dict(zip(graph, node_labels)), 'label')
    graph.graph['label'] = graph_labels
    return graph







