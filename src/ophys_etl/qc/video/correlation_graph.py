import argschema
import h5py
import itertools
import time
import networkx as nx
import numpy as np
import multiprocessing
from scipy.spatial.distance import cdist
from pathlib import Path
from typing import List

from ophys_etl.qc.video.schemas import CorrelationGraphInputSchema


def index_split(nelem: int, n_segments: int) -> List[List]:
    """splits elements into multiple ranges, with an overlap of 1
    These indices will be used in a numpy x[a:b] way, such that
    specifying [a, b] will result in indices a -> (b - 1)
    """
    indices = [[i.min(), i.max()] for i in
               np.array_split(np.arange(nelem), n_segments)]
    for index in indices:
        index[1] += 1
    for index in indices[1:]:
        index[0] -= 1
    return indices


def weight_calculation(video_path: Path, row_indices: List[int],
                       col_indices: List[int]) -> nx.Graph:
    """produces a graph with edges weighted according to
    Pearson's correlation coefficient, and nodes labeled according as
    (row_index, col_index).
    Each pixel is connected to 8 nearest-neighbors.
    """

    with h5py.File(video_path, "r") as f:
        data = f["data"][:,
                         row_indices[0]: row_indices[1],
                         col_indices[0]: col_indices[1]]

    # relative indices of 8 nearest-neighbors
    nbrs = list(itertools.product([-1, 0, 1], repeat=2))
    nbrs.pop(nbrs.index((0, 0)))

    nrow = data.shape[1]
    ncol = data.shape[2]
    r0 = row_indices[0]
    c0 = col_indices[0]

    graph = nx.Graph()
    for irow in range(nrow):
        for icol in range(ncol):
            edge_start = (irow, icol)
            global_edge_start = (irow + r0, icol + c0)
            edge_ends = []
            global_edges = []
            for dx, dy in nbrs:
                # a local edge, appropriate for indexing the loaded data
                edge_end = (irow + dy, icol + dx)
                if (edge_end[0] < 0) | (edge_end[0] >= nrow):
                    continue
                if (edge_end[1] < 0) | (edge_end[1] >= ncol):
                    continue
                # a global edge, appropriate for the entire FOV
                global_edge_end = (edge_end[0] + r0, edge_end[1] + c0)
                edge_ends.append(edge_end)
                global_edges.append([global_edge_start, global_edge_end])
            # cdist 2.5x faster for 1-to-many than repeated calls to pearsonr
            weights = 1.0 - cdist([data[:, edge_start[0], edge_start[1]]],
                                  [data[:, edge_end[0], edge_end[1]]
                                   for edge_end in edge_ends],
                                  metric="correlation")[0]
            for global_edge, weight in zip(global_edges, weights):
                if not graph.has_edge(*global_edge):
                    graph.add_edge(*global_edge, weight=weight)

    return graph


class CorrelationGraph(argschema.ArgSchemaParser):
    default_schema = CorrelationGraphInputSchema

    def run(self):
        self.logger.name = type(self).__name__
        t0 = time.time()

        with h5py.File(self.args["video_path"], "r") as f:
            nrow, ncol = f["data"].shape[1:]
        row_indices = index_split(nrow, self.args["n_segments"])
        col_indices = index_split(ncol, self.args["n_segments"])
        args = [(self.args["video_path"], rows, cols)
                for rows, cols in itertools.product(row_indices, col_indices)]

        if len(args) == 1:
            graph = weight_calculation(*args[0])
        else:
            self.logger.info(f"splitting into {len(args)} jobs")
            with multiprocessing.Pool(len(args)) as pool:
                graphs = pool.starmap(weight_calculation, args)
            self.logger.info(f"combining graphs from jobs")
            graph = nx.compose_all(graphs)

        nx.write_gpickle(graph, self.args["graph_output"])
        self.logger.info(f"wrote {self.args['graph_output']}")
        self.logger.info(f"finished in {time.time() - t0:2f} seconds")


if __name__ == "__main__":
    cg = CorrelationGraph()
    cg.run()
