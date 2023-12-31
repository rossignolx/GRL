import os
import random

from typing import Iterator, Dict
from itertools import combinations

import networkx
import networkx as nx

from labelling.label_graph import *

class GraphGenerator:
    def __init__(self):
        self.graph_counter = 0

    def generate_graph(
            self,
            method: str,
            no_nodes: int,
            params: Dict,
    ) -> nx.Graph:
        self.graph_counter += 1
        if method == 'empty':
            return nx.empty_graph(n=no_nodes)
        elif method == 'erdos':
            p = params.get("p", 0.5)
            return nx.fast_gnp_random_graph(no_nodes, p)
        elif method == 'connected':
            return nx.complete_graph(no_nodes)
        elif method == 'triangular_grid':
            return nx.triangular_lattice_graph(
                no_nodes, no_nodes, with_positions=False)
        elif method == 'special':
            triangle = nx.cycle_graph(3)
            r = np.random.choice(range(1 + no_nodes + 1))
            line = nx.path_graph(r)
            return nx.disjoint_union(triangle, line)
        elif method == 'half-special-half-connected':
            triangle = nx.cycle_graph(3)
            triangle_free = triangle_free_graph(no_nodes - 3)

            if self.graph_counter % 2 == 0:
                return nx.disjoint_union(triangle, triangle_free)
            else:
                union = nx.disjoint_union(triangle, triangle_free)
                for node in triangle_free:
                    node_id = node + 3
                    union.add_edge(0, node_id)
                return union
        else:
            raise ValueError()

def triangle_free_graph(total_nodes):
    g = nx.Graph()
    for i in range(total_nodes):
        g.add_node(i)

    random_edges = list(combinations(range(total_nodes), 2))
    random.shuffle(random_edges)

    for (u, v) in random_edges:
        set_nu = set(g.neighbors(u))
        set_nv = set(g.neighbors(v))
        if len(set_nu.intersection(set_nv)) == 0:
            g.add_edge(u, v)
    assert sum(list(nx.triangles(g).values())) == 0
    return g


def write_graphs(
        split: str,
        min_graphs: int,
        min_nodes: int,
        max_nodes: int,
        graph_method: str,
        formula: str,
        params: Dict,
        data_dir: str,
):
    graph_generator  = GraphGenerator()
    total_nodes = 0
    total_1s = 0
    total_graphs = 0
    total_graph_1s = 0
    total_labeled = 0

    # is_special = 'special' in graph_method or 'connected' in graph_method
    # prev_labeled = False

    graphs = []
    possible_nodes = range(min_nodes, max_nodes + 1)
    while True:
        min_graphs_achieved = total_graphs > min_graphs
        if min_graphs_achieved:
            break
            # node_balanced = (abs((total_1s / total_nodes) - 0.5) < 0.05)
            # if node_balanced:
            #     break

        no_nodes = random.choice(possible_nodes)
        graph = graph_generator.generate_graph(
            graph_method,
            no_nodes,
            params,
        )
        graph = networkx.convert_node_labels_to_integers(graph)

        # Empty graph
        if not list(graph.edges()):
            continue

        if formula == 'formula1':
            node_labels, graph_labels = connected_to_triangle_not_neighbour(graph)
        else:
            raise ValueError()

        # Balancing Logic. Quite ugly.
        # if not is_special:
        #     if not min_graphs_achieved:
        #         if prev_labeled:
        #             if graph_labels == 1:
        #                continue
        #             else:
        #                 prev_labeled = False
        #         else:
        #             if graph_labels == 0:
        #                 continue
        #             else:
        #                 percent = sum(node_labels) / len(node_labels)
        #                 if percent < 0.8:
        #                     continue
        #                 prev_labeled = True
        #
        #     else:
        #         percent = sum(node_labels) / len(node_labels)
        #         if percent < 0.8:
        #             continue

        if graph_labels == 1:
            total_labeled += 1

        label_graph(graph, node_labels, graph_labels)
        graphs.append(graph)

        total_graphs += 1
        total_nodes += no_nodes
        total_1s += sum(node_labels)
        total_graph_1s += graph_labels

        if total_graphs % 200 == 0:
            print("Generated {} graphs...".format(total_graphs))

    print("Total Nodes: {}, Total Node 1s: {}, Percentage 1s: {}%".format(total_nodes, total_1s, (total_1s / total_nodes) * 100))
    print("Total Graphs: {}, Total Graph 1s: {}, Percentage 1s: {}%".format(total_graphs, total_graph_1s, (total_graph_1s / total_graphs) * 100))
    print("Total Labeled Graphs: {}".format(total_labeled))

    file_path = os.path.join(data_dir, formula, split)
    os.makedirs(file_path, 0o777, exist_ok=True)

    full_path = os.path.join(file_path, "data.txt")

    with open(full_path, 'w') as f:
        f.write("{}\n".format(total_graphs))

        for graph in graphs:
            f.write("{} {}\n".format(len(graph), graph.graph['label']))
            for node in graph.nodes(data=True):
                idx, attrs = node
                edges = " ".join(map(str, list(graph[idx].keys())))
                no_edges = len(graph[idx])

                label = attrs['label']
                f.write("{} {} {}\n".format(label, no_edges, edges))

NO_NODES = 20
write_graphs(
    'train',
    2000,
    NO_NODES, NO_NODES,
    'erdos',
    'formula1',
    {'p': 0.15},
    "data/"
)

write_graphs(
    'val',
    500,
    NO_NODES, NO_NODES,
    'half-special-half-connected',
    'formula1',
    {},
    "data/"
)

write_graphs(
    'test',
    500,
    NO_NODES, NO_NODES,
    'half-special-half-connected',
    'formula1',
    {},
    "data/"
)
