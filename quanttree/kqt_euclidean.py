from typing import Union, List

import numpy as np

from quanttree.kqt_base import NodeFunction, PartitioningNode, NodeBasedPartitioning, TemplateQuantTree
from quanttree.kqt_utils import optimize_metric


class EuclideanKernel(NodeFunction):
    def __init__(self, centroid=None):
        super().__init__()
        self.centroid = centroid

    def __call__(self, data: np.ndarray, centroid: np.ndarray = None):
        if centroid is None:
            centroid = self.centroid
        return np.linalg.norm(data - centroid, axis=-1)


class EuclideanKernelNode(PartitioningNode):
    def __init__(self,
                 centroid: np.ndarray = None,
                 centroid_selection_metric: str = 'random',
                 centroid_selection_criterion: str = 'max',
                 centroid_selection_picks: int = 250):
        self.mapping: EuclideanKernel = EuclideanKernel(centroid=centroid)
        super(EuclideanKernelNode, self).__init__(mapping=self.mapping, direction='low')
        self.centroid: np.ndarray = centroid
        self.centroid_selection_metric: str = centroid_selection_metric
        self.centroid_selection_criterion: str = centroid_selection_criterion
        self.centroid_selection_picks: int = centroid_selection_picks

    def train(self, training_data: np.ndarray, percentage: float, do_transformation_fitting: bool = True):
        # part 1: estimate the best centroid
        ntraining_data, data_dimension = training_data.shape
        npartition = np.round(ntraining_data * percentage).astype(int)

        # centroid choice
        if self.centroid_selection_metric == 'fixed':
            assert self.centroid is not None, "No centroid provided"
            assert self.centroid.shape[-1] == data_dimension, "Invalid centroid provided"
        if self.centroid_selection_metric == 'random':
            self.mapping.centroid = training_data[np.random.choice(ntraining_data)]
        else:
            self.mapping.centroid, _ = optimize_metric(quality_metric=self.centroid_selection_metric,
                                                       minmax=self.centroid_selection_criterion,
                                                       data=training_data,
                                                       kernel=self.mapping,
                                                       num_inbin=npartition,
                                                       npicks=self.centroid_selection_picks)
        self.centroid = self.mapping.centroid

        # part 2: train the node with the now-fixed mapping
        return super().train(
            training_data=training_data,
            percentage=percentage
        )


class EuclideanKernelPartitioning(NodeBasedPartitioning):
    def __init__(self, pi_values: Union[int, List[float], np.ndarray], transformation_type: str = 'none',
                 centroid: np.ndarray = None, centroid_selection_metric: str = 'random',
                 centroid_selection_criterion: str = 'max', centroid_selection_picks: int = 250):
        super().__init__(
            pi_values=pi_values,
            transformation_type=transformation_type
        )

        self.training_distribution: np.ndarray = self.pi_values

        self.nodes: List[EuclideanKernelNode] = []
        self.centroid: np.ndarray = centroid
        self.centroid_selection_metric: str = centroid_selection_metric
        self.centroid_selection_criterion: str = centroid_selection_criterion
        self.centroid_selection_picks: int = centroid_selection_picks

    def _build_partitioning(self, training_data: np.ndarray):
        # training_data: (ntraining_points, data_dimension)
        self.ntraining_data, self.data_dimension = training_data.shape

        self.nodes: List[EuclideanKernelNode] = []
        self.training_distribution = np.zeros(self.nbins)

        for ibin in range(self.nbins - 1):
            if training_data.shape[0] == 0:
                break

            target = self.pi_values[ibin] / self.pi_values[ibin:].sum()
            self.nodes.append(EuclideanKernelNode(
                centroid=self.centroid,
                centroid_selection_metric=self.centroid_selection_metric,
                centroid_selection_criterion=self.centroid_selection_criterion,
                centroid_selection_picks=self.centroid_selection_picks
            ))

            in_bin = self.nodes[ibin].train(training_data=training_data, percentage=target)
            self.training_distribution[ibin] = np.sum(in_bin)

            training_data = np.delete(training_data, in_bin, axis=0)

        self.training_distribution[-1] = training_data.shape[0]

        assert np.sum(self.training_distribution).astype(int) == self.ntraining_data, \
            f"{np.sum(self.training_distribution).astype(int)} =/= {self.ntraining_data}"

    def _find_bin(self, data: np.ndarray):
        return super()._find_bin(data=data)

    def _get_bin_counts(self, data: np.ndarray):
        return super()._get_bin_counts(data=data)


class EuclideanKernelQuantTree(TemplateQuantTree):
    def __init__(self, pi_values: Union[int, List[float], np.ndarray], nu: int, alpha: float,
                 statistic_name: str = 'pearson', transformation_type: str = 'none', nbatch: int = 100000,
                 centroid: np.ndarray = None,
                 centroid_selection_metric: str = 'random', centroid_selection_criterion='max',
                 centroid_selection_picks: int = 250):
        self.partitioning = EuclideanKernelPartitioning(
            pi_values=pi_values,
            transformation_type=transformation_type,
            centroid=centroid,
            centroid_selection_metric=centroid_selection_metric,
            centroid_selection_criterion=centroid_selection_criterion,
            centroid_selection_picks=centroid_selection_picks
        )
        super().__init__(
            partitioning=self.partitioning,
            statistic_name=statistic_name,
            pi_values=self.partitioning.pi_values,
            nu=nu, alpha=alpha,
            transformation_type=transformation_type,
            nbatch=nbatch
        )
        self.pi_values = self.partitioning.pi_values
