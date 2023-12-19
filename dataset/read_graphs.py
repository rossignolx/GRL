from typing import Union, List, Tuple

import networkx as nx
import shutil
import torch
import os
from torch_geometric.data import InMemoryDataset, Data

class GraphLogicDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super(GraphLogicDataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        return ['data.txt']

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return 'data.pt'

    def process(self):
        data_list = []

        full_path = os.path.join(self.raw_dir, 'data.txt')
        with open(full_path, 'r') as f:
            no_graphs = int(f.readline().strip())
            for _ in range(no_graphs):
                no_nodes = int(f.readline().strip().split(" ")[0])

                node_labels = []
                graph = nx.DiGraph()

                for id in range(no_nodes):
                    graph.add_node(id)
                    info = list(int(x) for x in f.readline().strip().split(" "))

                    label = info[0]
                    node_labels.append(label)

                    neighbours = info[2:]
                    for neighbour in neighbours:
                        graph.add_edge(id, neighbour)
                        graph.add_edge(neighbour, id)

                node_labels = torch.tensor(node_labels, dtype=torch.long)

                # Constant features, for now.
                edges = torch.tensor(list(graph.edges), dtype=torch.long)
                node_features = torch.ones(no_nodes).view(-1, 1)

                data = Data(
                        x=node_features,
                        edge_index=edges.t().contiguous(),
                        node_labels=node_labels,
                    )
                data_list.append(data)

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def download(self):
        r, variant = os.path.split(self.root)
        r, split = os.path.split(r)
        r, formula_name = os.path.split(r)

        shutil.copy('data/{}/{}/data.txt'.format(formula_name, split), self.raw_dir)
