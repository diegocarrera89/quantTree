from typing import Union, List

import numpy as np
from sklearn.covariance import MinCovDet
from quanttree.kqt_base import NodeFunction, PartitioningNode, NodeBasedPartitioning, TemplateQuantTree
from quanttree.kqt_utils import optimize_metric


class MahalanobisKernel(NodeFunction):
    def __init__(self,
                 cov: np.ndarray = None,
                 invcov: np.ndarray = None,
                 centroid: np.ndarray = None,
                 do_robust_covariance: bool = False
                 ):
        super().__init__()
        if cov is None:
            self.cov = None
            self.invcov = None
        else:
            self.cov = cov
            if invcov is None:
                self.invcov = np.linalg.inv(cov)
            else:
                self.invcov = invcov
        self.centroid: np.ndarray = centroid
        self.do_robust_covariance: bool = do_robust_covariance

    def __call__(self, data: np.ndarray, centroid: np.ndarray = None):
        assert self.cov is not None, "Mahalanobis Kernel not initialized. Call 'fit' before using the node."
        if centroid is None:
            centroid = self.centroid
        translated = data - centroid
        result = np.matmul(translated, self.invcov)
        result = np.sum(result * translated, axis=1) ** .5
        return result

    def fit(self, training_data: np.ndarray):
        if self.cov is None:
            if self.do_robust_covariance:
                cov_est = MinCovDet(support_fraction=.95)
                cov_est.fit(training_data)
                self.cov = cov_est.covariance_
            else:
                self.cov = np.cov(training_data, rowvar=False)
            self.invcov = np.linalg.inv(self.cov)
        return self


class MahalanobisNode(PartitioningNode):
    def __init__(self, centroid: np.ndarray = None, centroid_selection_metric: str = 'random',
                 centroid_selection_criterion='max', centroid_selection_picks: int = 250,
                 do_robust_covariance: bool = False, global_covariance: np.ndarray = None):
        self.mapping: MahalanobisKernel = MahalanobisKernel()
        super(MahalanobisNode, self).__init__(
            mapping=self.mapping,
            direction='low'
        )

        self.mapping = MahalanobisKernel(cov=global_covariance, do_robust_covariance=do_robust_covariance)
        self.centroid: np.ndarray = centroid
        self.centroid_selection_metric: str = centroid_selection_metric
        self.centroid_selection_criterion: str = centroid_selection_criterion
        self.centroid_selection_picks: int = centroid_selection_picks

    def train(self, training_data: np.ndarray, percentage: float, do_transformation_fitting: bool = True):
        ntraining_data, data_dimension = training_data.shape
        npartition = np.round(ntraining_data * percentage).astype(int)

        self.mapping.fit(training_data=training_data)

        # centroid choice
        if self.centroid_selection_metric == 'fixed':
            assert self.centroid is not None, "No centroid provided"
            assert self.centroid.shape[-1] == data_dimension, "Invalid centroid provided"
        elif self.centroid_selection_metric == 'random':
            self.mapping.centroid = training_data[np.random.choice(ntraining_data)]
        else:
            self.mapping.centroid, _ = optimize_metric(quality_metric=self.centroid_selection_metric,
                                                       minmax=self.centroid_selection_criterion,
                                                       data=training_data,
                                                       kernel=self.mapping,
                                                       num_inbin=npartition,
                                                       npicks=self.centroid_selection_picks)

        self.centroid = self.mapping.centroid

        return super().train(
            training_data=training_data,
            percentage=percentage
        )


class MahalanobisPartitioning(NodeBasedPartitioning):
    def __init__(self, pi_values: Union[int, List[float], np.ndarray], transformation_type: str = 'none',
                 centroid: np.ndarray = None, centroid_selection_metric: str = 'random',
                 centroid_selection_criterion='max', centroid_selection_picks: int = 250,
                 do_robust_covariance: bool = False, use_global_covariance: bool = False):
        super().__init__(
            pi_values=pi_values,
            transformation_type=transformation_type
        )

        self.nodes: List[MahalanobisNode] = []
        self.training_distribution: np.ndarray = np.zeros(self.nbins)
        self.centroid: np.ndarray = centroid
        self.centroid_selection_metric: str = centroid_selection_metric
        self.centroid_selection_criterion: str = centroid_selection_criterion
        self.centroid_selection_picks: int = centroid_selection_picks
        self.do_robust_covariance: bool = do_robust_covariance
        self.use_global_covariance: bool = use_global_covariance
        self.global_covariance = None
        assert not (self.use_global_covariance and self.transformation_type != 'none')

    def _build_partitioning(self, training_data: np.ndarray):
        # training_data: (ntraining_points, data_dimension)
        self.ntraining_data, self.data_dimension = training_data.shape

        self.nodes: List[MahalanobisNode] = []
        self.training_distribution = np.zeros(self.nbins)

        if self.use_global_covariance:
            self.global_covariance = np.cov(training_data, rowvar=False)

        for ibin in range(self.nbins - 1):
            if training_data.shape[0] == 0:
                break

            target = self.pi_values[ibin] / self.pi_values[ibin:].sum()
            self.nodes.append(
                MahalanobisNode(centroid=self.centroid, centroid_selection_metric=self.centroid_selection_metric,
                                centroid_selection_criterion=self.centroid_selection_criterion,
                                centroid_selection_picks=self.centroid_selection_picks,
                                do_robust_covariance=self.do_robust_covariance,
                                global_covariance=self.global_covariance))

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


class MahalanobisKernelQuantTree(TemplateQuantTree):
    def __init__(self, pi_values: Union[int, List[float], np.ndarray], nu: int, alpha: float,
                 statistic_name: str = 'pearson', transformation_type: str = 'none', nbatch: int = 100000,
                 centroid: np.ndarray = None,
                 centroid_selection_metric: str = 'random', centroid_selection_criterion='max',
                 centroid_selection_picks: int = 250, do_robust_covariance: bool = False,
                 use_global_covariance: bool = False):
        self.partitioning = MahalanobisPartitioning(pi_values=pi_values, transformation_type=transformation_type,
                                                    centroid=centroid,
                                                    centroid_selection_metric=centroid_selection_metric,
                                                    centroid_selection_criterion=centroid_selection_criterion,
                                                    centroid_selection_picks=centroid_selection_picks,
                                                    do_robust_covariance=do_robust_covariance,
                                                    use_global_covariance=use_global_covariance)
        super().__init__(
            partitioning=self.partitioning,
            statistic_name=statistic_name,
            pi_values=self.partitioning.pi_values,
            nu=nu, alpha=alpha,
            transformation_type=transformation_type,
            nbatch=nbatch
        )
        self.pi_values = self.partitioning.pi_values
