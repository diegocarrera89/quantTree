from typing import Union, List
import numpy as np
from quanttree.kqt_base import NodeFunction, PartitioningNode, NodeBasedPartitioning, TemplateQuantTree


class Projection(NodeFunction):
    def __init__(self, dimension: int):
        """
        Projection function over the dimension taken as input

        :param dimension: dimension over which to project data
        """
        super().__init__()
        self.dimension = dimension

    def __call__(self, data: np.ndarray):
        """
        Projects data over the chosen dimension

        :param data: np.ndarray, data to be projected with shape (npoints, dimensions)
        :return: projected data
        """
        return data[:, self.dimension]

    def __repr__(self):
        return f"Projection(dimension = {self.dimension})"


class ProjectionNode(PartitioningNode):
    def __init__(self, proj_dimension: int = -1,
                 direction: str = 'random'):
        super().__init__(
            mapping=Projection(dimension=proj_dimension),
            direction=direction
        )

    def train(self, training_data: np.ndarray, percentage: float, do_transformation_fitting: bool = True):
        if self.mapping.dimension == -1:
            self.mapping.dimension = np.random.randint(low=0, high=training_data.shape[-1])

        return super(ProjectionNode, self).train(
            training_data=training_data,
            percentage=percentage
        )


class NodeQuantTreePartitioning(NodeBasedPartitioning):
    def __init__(self, pi_values: Union[int, List[float], np.ndarray],
                 transformation_type: str = 'none'):
        super().__init__(
            pi_values=pi_values,
            transformation_type=transformation_type
        )

        self.nodes: List[ProjectionNode] = []
        self.training_distribution: np.ndarray = np.zeros(self.nbins)

    def _build_partitioning(self, training_data: np.ndarray):
        # training_data: (ntraining_points, data_dimension)
        self.ntraining_data, self.data_dimension = training_data.shape

        self.nodes: List[ProjectionNode] = []
        self.training_distribution = np.zeros(self.nbins)

        for ibin in range(self.nbins - 1):
            if training_data.shape[0] == 0:
                break

            target = self.pi_values[ibin] / self.pi_values[ibin:].sum()
            self.nodes.append(ProjectionNode())

            in_bin = self.nodes[ibin].train(training_data=training_data, percentage=target)
            self.training_distribution[ibin] = np.sum(in_bin)

            training_data = np.delete(training_data, in_bin, axis=0)

        self.training_distribution[-1] = training_data.shape[0]

        assert np.sum(self.training_distribution).astype(int) == self.ntraining_data,\
            f"{np.sum(self.training_distribution).astype(int)} =/= {self.ntraining_data}"

    def _find_bin(self, data: np.ndarray):
        return super()._find_bin(data=data)

    def _get_bin_counts(self, data: np.ndarray):
        return super()._get_bin_counts(data=data)


class NodeQuantTree(TemplateQuantTree):
    def __init__(self,
                 pi_values: Union[int, List[float], np.ndarray],
                 nu: int,
                 alpha: float,
                 statistic_name: str = 'pearson',
                 transformation_type: str = 'none',
                 nbatch: int = 100000):
        self.partitioning = NodeQuantTreePartitioning(
            pi_values=pi_values,
            transformation_type=transformation_type
        )
        super().__init__(
            partitioning=self.partitioning,
            statistic_name=statistic_name,
            pi_values=self.partitioning.pi_values,
            nu=nu, alpha=alpha,
            transformation_type=transformation_type,
            nbatch=nbatch
        )
        self.pi_values: np.ndarray = self.partitioning.pi_values
