import warnings
from abc import ABC, abstractmethod
from typing import Union, List

import numpy as np
from quanttree.core import DataTransformation, Statistic
from quanttree import QuantTreeThresholdStrategy


class NodeFunction:
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs) -> np.ndarray:
        pass

    def fit(self, training_data: np.ndarray):
        pass


class PartitioningNode(ABC):
    def __init__(self, mapping: NodeFunction,
                 direction: str = 'low'):
        """
        Splits the data into in-bin data and out-of-bin data

        :param mapping: mapping from R^d to R (required to sort data)
        :param direction: determines whether high-valued samples or low-valued samples are in-bin
        """
        self.mapping: NodeFunction = mapping
        self.threshold: float = 0.0
        self.direction: str = direction

        if self.direction not in ['low', 'high', 'random']:
            warnings.warn(f"Invalid direction provided: {direction} \nUsing 'low'")
            self.direction = 'low'
        if self.direction == 'random':
            if np.random.randn() > 0:
                self.direction = 'high'
            else:
                self.direction = 'low'

    def __call__(self, data: np.ndarray):
        """
        Transform data, map it to R and compare the output with the threshold

        :param data: data to be split
        :return: bool array where True means in-bin
        """
        scores = self.mapping(data)
        if self.direction == 'low':
            return scores <= self.threshold
        if self.direction == 'high':
            return scores >= self.threshold
        warnings.warn("Direction has not been set.")
        return None

    @abstractmethod
    def train(self, training_data: np.ndarray, percentage: float):
        """
        Fits the transformatoin and sets the threshold such that the desired percentage of points ends up in the bin

        Parameters
        ----------
        training_data : np.ndarray
            Training data with shape (nsamples, dim)
        percentage : float
            Percentage of points to be kept in the bin

        Returns
        -------
        np.ndarray[bools] (nsamples,)
            Whether the sample is in-bin or not
        """
        nsamples = training_data.shape[0]
        npartition = int(nsamples * percentage)

        scores = self.mapping(training_data)
        sorted_scores = np.sort(scores)

        if self.direction == 'low':
            self.threshold = sorted_scores[npartition - 1]
            return scores <= self.threshold
        if self.direction == 'high':
            self.threshold = sorted_scores[-npartition]
            return scores >= self.threshold


class NodeBasedPartitioning(ABC):
    def __init__(self, pi_values: Union[int, List[float], np.ndarray], transformation_type: str = 'none'):
        assert type(pi_values) in [int, List[float], np.ndarray], "Invalid type for pi_values"
        if type(pi_values) == int:
            self.nbins: int = pi_values
            self.pi_values: np.ndarray = np.ones(self.nbins) / self.nbins
        if type(pi_values) == List[float]:
            self.nbins: int = len(pi_values)
            self.pi_values: np.ndarray = np.array(pi_values)
        if type(pi_values) == np.ndarray:
            assert len(pi_values.shape) == 1
            self.nbins: int = pi_values.shape[0]
            self.pi_values: np.ndarray = pi_values / pi_values.sum()

        self.transformation_type: str = transformation_type
        self.transformation: DataTransformation = DataTransformation.get_data_transformation(transformation_type)

        self.nodes: List[PartitioningNode] = []
        self.ntraining_data: int = 0
        self.data_dimension: int = 0
        self.training_distribution = np.zeros(self.nbins)

    @abstractmethod
    def _build_partitioning(self, training_data: np.ndarray):
        assert len(training_data.shape) == 2
        self.nodes: List[PartitioningNode] = []
        self.ntraining_data, self.data_dimension = training_data.shape

        for ibin in range(self.nbins - 1):
            self.nodes.append(PartitioningNode(mapping=NodeFunction()))

            in_bin = self.nodes[ibin].train(training_data=training_data,
                                            percentage=self.pi_values[ibin] / self.pi_values[ibin:].sum())

            self.training_distribution[ibin] = np.sum(in_bin)

            training_data = np.delete(training_data, in_bin)

    @abstractmethod
    def _find_bin(self, data: np.ndarray):
        # data: (nbatches, nu, dim) or (ndata, dim)
        nbatches, nu = -1, -1
        if len(data.shape) not in [2, 3]:
            raise ValueError("Data must be 2- or 3-dimensional")
        if len(data.shape) == 2:
            nbatches = 1
            nu, dim = data.shape
        if len(data.shape) == 3:
            nbatches, nu, dim = data.shape
            data = data.reshape((nbatches * nu, dim))
        ndata_points = nbatches * nu

        bins = np.ones(ndata_points) * (self.nbins - 1)
        available = np.arange(ndata_points)
        for ibin, node in enumerate(self.nodes):
            if available.shape[0] == 0:
                break
            in_bin = node(data[available])
            bins[available[in_bin]] = ibin
            available = np.delete(available, in_bin, axis=0)

        return bins.reshape((nbatches, nu)).squeeze().astype(int)

    @abstractmethod
    def _get_bin_counts(self, data: np.ndarray):
        bin_indexes = self._find_bin(data)
        if len(bin_indexes.shape) == 1:
            return np.bincount(bin_indexes, minlength=self.nbins)
        if len(bin_indexes.shape) == 2:
            return np.array([np.bincount(batch_indexes, minlength=self.nbins) for batch_indexes in bin_indexes]).astype(int)

    def build_partitioning(self, training_data: np.ndarray):
        self.transformation.estimate_transformation(data=training_data)
        self._build_partitioning(training_data=self.transformation.transform_data(data=training_data))

    def find_bin(self, data: np.ndarray):
        return self._find_bin(data=self.transformation.transform_data(data=data))

    def get_bin_counts(self, data: np.ndarray):
        return self._get_bin_counts(data=self.transformation.transform_data(data=data))


class NodeBasedHistogram(ABC):
    def __init__(self, partitioning: NodeBasedPartitioning, statistic_name: str = 'pearson'):
        self.partitioning: NodeBasedPartitioning = partitioning

        # Statistic name and instance
        self.statistic_name: str = statistic_name
        self.statistic: Statistic = Statistic()

        # Uniformly initialized pi_values
        # self.pi_values: np.ndarray = np.ones(self.partitioning.nbins) * 1 / self.partitioning.nbins
        self.pi_values: np.ndarray = self.partitioning.pi_values
        self.threshold: float = 0.0

    @abstractmethod
    def train_model(self, training_data: np.ndarray):
        self.partitioning.build_partitioning(training_data=training_data)
        self.set_statistic()

    def assess_goodness_of_fit(self, data: np.ndarray):
        assert len(data.shape) in [2, 3], f"Invalid data shape: {data.shape}"
        nu = 0
        if len(data.shape) == 2:
            nu, _ = data.shape
            data = data.reshape((1,) + data.shape)
        if len(data.shape) == 3:
            nbatches, nu, _ = data.shape

        pi_hats = self.partitioning.get_bin_counts(data) / nu
        if len(pi_hats.shape) == 1:
            pi_hats = pi_hats.reshape((1,) + pi_hats.shape)
        return self.statistic.compute_statistic(pi_hats=pi_hats, nu=nu)

    def get_bin_counts(self, data: np.ndarray):
        return self.partitioning.get_bin_counts(data)

    def set_statistic(self):
        self.statistic = Statistic(statistic_name=self.statistic_name, pi_values=self.pi_values)


class TemplateQuantTree(NodeBasedHistogram, ABC):
    def __init__(self,
                 partitioning: NodeBasedPartitioning,
                 statistic_name: str,
                 pi_values: Union[int, List[float], np.ndarray],
                 nu: int,
                 alpha: float,
                 transformation_type: str = 'none',
                 nbatch: int = 100000):
        super(TemplateQuantTree, self).__init__(partitioning=partitioning, statistic_name=statistic_name)

        self.pi_values = pi_values
        self.nu: int = nu
        self.alpha: float = alpha
        self.ndata_training: int = 0

        self.transformation_type: str = transformation_type

        self.nbatch: int = nbatch
        self.threshold: float = 0.0
        self.threshold_strat = None

    def train_model(self, training_data: np.ndarray, do_threshold: bool = True):
        self.ndata_training = training_data.shape[0]
        self.partitioning.build_partitioning(training_data=training_data)
        self.pi_values = self.partitioning.training_distribution / self.ndata_training
        self.set_statistic()
        if do_threshold:
            self.compute_threshold()

    def compute_threshold(self):
        self.threshold_strat = QuantTreeThresholdStrategy(nu=self.nu, ndata_training=self.ndata_training,
                                                          pi_values=self.pi_values, nbatch=self.nbatch,
                                                          statistic_name=self.statistic_name)

        self.threshold = self.threshold_strat.get_threshold(alpha=self.alpha)
