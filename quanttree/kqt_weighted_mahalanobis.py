from typing import Union, List

import numpy as np
from scipy.special import erf
from sklearn.mixture import BayesianGaussianMixture

from quanttree.kqt_base import NodeFunction, PartitioningNode, NodeBasedPartitioning, TemplateQuantTree
from quanttree.kqt_utils import optimize_metric


class WeightedMahalanobisKernel(NodeFunction):
    def __init__(
            self,
            weights: np.ndarray = None,
            means: np.ndarray = None,
            covs: np.ndarray = None,
            invcovs: np.ndarray = None,
            centroid: np.ndarray = None,
    ):
        super().__init__()

        if covs is None:
            self.weights = None
            self.means = None
            self.covs = None
            self.invcovs = None
        else:
            self.weights = weights
            self.means = means
            self.covs = covs
            if invcovs is None:
                self.invcovs = np.array([np.linalg.inv(cov) for cov in covs])
            else:
                self.invcovs = invcovs
        self.centroid: np.ndarray = centroid

    def __call__(self, data: np.ndarray, centroid: np.ndarray = None):
        assert self.covs is not None, "Mahalanobis Kernel not initialized. Call 'fit' before using the node."
        if centroid is None:
            centroid = self.centroid

        numerator = np.zeros(data.shape[0])
        denominator = np.zeros(data.shape[0])

        indexes_zero = np.where(np.sum(np.abs(data - centroid), axis=1) == 0)[0]

        v = data - centroid

        for k, (w_i, mu_i, invcov_i) in enumerate(zip(self.weights, self.means, self.invcovs)):
            u = mu_i - centroid
            vC = np.dot(v, invcov_i)

            vCv = np.einsum("nd, nd->n", vC, v)
            vCu = np.einsum("nd, d->n", vC, u)

            uCu = np.sum(u * np.matmul(invcov_i, u))

            # TODO: check fix
            vCv[np.where(vCv == 0)] = 1e-16
            # vCu[np.where(vCu == 0)] = 1e-16
            bsquared = 1 / vCv

            a = bsquared * vCu
            Z = uCu - bsquared * (vCu ** 2)
            factor = (2 * bsquared) ** .5

            integral = (np.pi * bsquared / 2) ** .5
            integral *= np.exp(-Z / 2)
            integral *= erf((1 - a) / factor) - erf(-a / factor)

            numerator += w_i * vCv * integral
            denominator += w_i * integral

        denominator += 1e-16
        dists = (numerator / denominator) ** .5

        dists[indexes_zero] = 0
        return dists

    def fit(self, training_data: np.ndarray):
        return self


class WeightedMahalanobisNode(PartitioningNode):
    def __init__(self, centroid: np.ndarray = None, centroid_selection_metric: str = 'random',
                 centroid_selection_criterion='max', centroid_selection_picks: int = 250,
                 weights: np.ndarray = None, means: np.ndarray = None,
                 covs: np.ndarray = None, invcovs: np.ndarray = None
                 ):
        self.mapping: WeightedMahalanobisKernel = WeightedMahalanobisKernel(weights=weights, means=means, covs=covs,
                                                                            invcovs=invcovs, centroid=centroid)
        super(WeightedMahalanobisNode, self).__init__(
            mapping=self.mapping,
            direction='low'
        )

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
            self.mapping.centroid = training_data[np.random.choice(training_data.shape[0])]
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


class WeightedMahalanobisPartitioning(NodeBasedPartitioning):
    def __init__(self, pi_values: Union[int, List[float], np.ndarray], transformation_type: str = 'none',
                 centroid: np.ndarray = None, centroid_selection_metric: str = 'random',
                 centroid_selection_criterion='max', centroid_selection_picks: int = 250,
                 bgm_num_components: int = 4,
                 bgm_init_params: str = "random",  # "kmeans"
                 bgm_max_iter: int = 8000,
                 bgm_mean_precision_prior: float = None,  # 0.8
                 bgm_random_state: int = None  # 42
                 ):
        super().__init__(
            pi_values=pi_values,
            transformation_type=transformation_type
        )

        self.nodes: List[WeightedMahalanobisNode] = []
        self.training_distribution: np.ndarray = np.zeros(self.nbins)
        self.centroid: np.ndarray = centroid
        self.centroid_selection_metric: str = centroid_selection_metric
        self.centroid_selection_criterion: str = centroid_selection_criterion
        self.centroid_selection_picks: int = centroid_selection_picks
        self.bgm_num_components: int = bgm_num_components
        self.bgm_init_params: str = bgm_init_params
        self.bgm_max_iter: int = bgm_max_iter
        self.bgm_mean_precision_prior: float = bgm_mean_precision_prior
        self.bgm_random_state: int = bgm_random_state

    def _build_partitioning(self, training_data: np.ndarray):
        # training_data: (ntraining_points, data_dimension)
        self.ntraining_data, self.data_dimension = training_data.shape

        self.nodes: List[WeightedMahalanobisNode] = []
        self.training_distribution = np.zeros(self.nbins)

        bgm = BayesianGaussianMixture(
            weight_concentration_prior_type="dirichlet_distribution",
            n_components=self.bgm_num_components,
            init_params=self.bgm_init_params,
            max_iter=self.bgm_max_iter,
            mean_precision_prior=self.bgm_mean_precision_prior,
            random_state=self.bgm_random_state
        )
        bgm.fit(training_data)

        weights = bgm.weights_
        means = bgm.means_
        covs = bgm.covariances_
        invcovs = np.array([np.linalg.inv(cov) for cov in covs])

        for ibin in range(self.nbins - 1):
            if training_data.shape[0] == 0:
                break

            target = self.pi_values[ibin] / self.pi_values[ibin:].sum()
            self.nodes.append(
                WeightedMahalanobisNode(centroid=self.centroid,
                                        centroid_selection_metric=self.centroid_selection_metric,
                                        centroid_selection_criterion=self.centroid_selection_criterion,
                                        centroid_selection_picks=self.centroid_selection_picks,
                                        weights=weights,
                                        means=means,
                                        covs=covs,
                                        invcovs=invcovs))

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


class WeightedMahalanobisKernelQuantTree(TemplateQuantTree):
    def __init__(self, pi_values: Union[int, List[float], np.ndarray], nu: int, alpha: float,
                 statistic_name: str = 'pearson', transformation_type: str = 'none', nbatch: int = 100000,
                 centroid: np.ndarray = None,
                 centroid_selection_metric: str = 'random', centroid_selection_criterion='max',
                 centroid_selection_picks: int = 250, bgm_num_components: int = 4):
        self.partitioning = WeightedMahalanobisPartitioning(pi_values=pi_values,
                                                            transformation_type=transformation_type,
                                                            centroid=centroid,
                                                            centroid_selection_metric=centroid_selection_metric,
                                                            centroid_selection_criterion=centroid_selection_criterion,
                                                            centroid_selection_picks=centroid_selection_picks,
                                                            bgm_num_components=bgm_num_components)
        super().__init__(
            partitioning=self.partitioning,
            statistic_name=statistic_name,
            pi_values=self.partitioning.pi_values,
            nu=nu, alpha=alpha,
            transformation_type=transformation_type,
            nbatch=nbatch
        )
        self.pi_values = self.partitioning.pi_values
