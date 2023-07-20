import os
import pickle
import warnings
from typing import Union

import numpy as np
from numpy.random import dirichlet
from scipy.stats import mvn

from quanttree.core import Partitioning, reshape_data, ThresholdStrategy, Histogram
from quanttree.utils import pkg_folder


# Partitioning
class QuantTreePartitioning(Partitioning):

    def __init__(self, pi_values: Union[int, list, np.ndarray] = 2, transformation_type: str = 'none'):
        pi_values = np.array(pi_values)
        if pi_values.size == 1:
            nbin = pi_values
            self.pi_values = np.ones(nbin) / nbin
            self.is_unif = True
        else:
            self.pi_values = np.array(pi_values)
            if len(np.unique(pi_values)) == 1:
                self.is_unif = True
            else:
                self.is_unif = False
            nbin = len(pi_values)

        super().__init__(nbin, transformation_type)
        self.leaves: np.ndarray = np.array([])  # MOD
        self.ndata_training = []
        self.dim = None

    def _build_partitioning(self, data):
        data = reshape_data(data)
        if len(data.shape) > 2:
            data = data.reshape(-1, data.shape[-1])
        ndata, ndim = data.shape
        nbin = len(self.pi_values)

        self.ndata_training = ndata
        self.dim = ndim

        # Each leaf is characterized by 4 numbers:
        # 1) the dimension of the split that generates the leaf,
        # 2) the lower bound of the leaf,
        # 3) the upper bound of the leaf,
        # 4) the cut direction
        self.leaves = np.ones(shape=(4, nbin))

        # set the limits of the available space in each dimension
        limits = np.ones((2, ndim))
        limits[0, :] = -np.inf
        limits[1, :] = np.inf

        # all samples are available
        available = [True] * ndata

        # iteratively generate the leaves
        for i_leaf in range(nbin - 1):
            # select a random components
            i_dim = np.random.randint(ndim)
            x_tilde = data[available, i_dim]

            # find the indices of the available samples
            # idx = [i for i in range(len(available)) if available[i]]
            idx = np.where(available)[0]
            N_tilde = len(idx)

            # sort the samples
            # idx_sorted = sorted(range(len(x_tilde)), key=x_tilde.__getitem__)
            # x_tilde.sort()
            idx_sorted = np.argsort(x_tilde)
            x_tilde = x_tilde[idx_sorted]

            # compute p_tilde
            p_tilde = self.pi_values[i_leaf] / (1 - np.sum(self.pi_values[0:i_leaf]))
            L = int(np.round(p_tilde * N_tilde))

            # define the leaf
            if np.random.choice([True, False]):
                self.leaves[:, i_leaf] = [i_dim, limits[0, i_dim], x_tilde[L - 1], -1]
                limits[0, i_dim] = x_tilde[L - 1]
                idx_sorted = idx_sorted[0:L]
            else:
                self.leaves[:, i_leaf] = [i_dim, x_tilde[-L], limits[1, i_dim], 1]
                limits[1, i_dim] = x_tilde[-L]
                idx_sorted = idx_sorted[-L:]

            # remove the sample in the leaf from the available samples
            for i in idx_sorted:
                available[idx[i]] = False

        # define the last leaf with the remaining samples
        i_dim = np.random.randint(ndim)
        self.leaves[:, -1] = [i_dim, limits[0, i_dim], limits[1, i_dim], 0]

    def _find_bin(self, data):
        # Expected data shape: ([nbatches], nu, dim)
        if len(data.shape) == 2:
            nbatches = 1
            nu, dim = data.shape
        else:
            nbatches, nu, dim = data.shape

        npoints = nbatches * nu
        data = data.reshape((npoints, dim))
        leaves = np.ones(npoints).astype(int) * (self.nbin - 1)
        avidxs = np.arange(npoints)
        for i_leaf in range(self.nbin - 1):
            ldim = self.leaves[0, i_leaf].astype(int)
            x_tilde = data[avidxs, ldim]
            # Get the indexes of the points falling in bin[i_leaf]
            if self.leaves[3, i_leaf] == -1:
                binidxs = np.where((self.leaves[1, i_leaf] < x_tilde) & (x_tilde <= self.leaves[2, i_leaf]))
            elif self.leaves[3, i_leaf] == 1:
                binidxs = np.where((self.leaves[1, i_leaf] <= x_tilde) & (x_tilde < self.leaves[2, i_leaf]))
            else:
                binidxs = np.where((self.leaves[1, i_leaf] < x_tilde) & (x_tilde < self.leaves[2, i_leaf]))

            # Update
            leaves[avidxs[binidxs]] = i_leaf
            avidxs = np.delete(avidxs, binidxs, 0)
            if len(avidxs) == 0:
                break

        # the squeeze function is to ensure that 2-dimensional inputs have 2-dimensional outputs
        return leaves.reshape((nbatches, nu)).squeeze()

    def _get_bin_counts(self, data):
        # data: ([nbathces], nu, dim)
        # bins: ([nbatches], nu)
        # bin_counts: ([nbatches], K)
        bins = self._find_bin(data)
        if len(bins.shape) == 1:
            bins = bins.reshape((1,) + bins.shape)
        # bin_counts = np.array([np.bincount(babin, minlength=self.nbin) for babin in bins])
        return np.array([np.bincount(babin, minlength=self.nbin) for babin in bins]).squeeze()

    def get_leaves_box(self):
        nleaves = self.leaves.shape[1]
        dim = self.dim
        box = np.ndarray(shape=(2, dim, nleaves))

        box[0, :, :] = -np.inf
        box[1, :, :] = np.inf

        for i_leaf in range(nleaves):
            dim_split = int(self.leaves[0, i_leaf])
            # controllo se lo split e' stato fatto prendendo la coda destra o sinistra
            is_lower_split = box[0, dim_split, i_leaf] == self.leaves[1, i_leaf]

            # calcolo il box
            box[0, dim_split, i_leaf] = self.leaves[1, i_leaf]
            box[1, dim_split, i_leaf] = self.leaves[2, i_leaf]

            # aggiorno i limiti dei box successivi
            if is_lower_split:
                box[0, dim_split, i_leaf + 1:] = box[1, dim_split, i_leaf]
            else:
                box[1, dim_split, i_leaf + 1:] = box[0, dim_split, i_leaf]

        return box

    def compute_gaussian_probabilities(self, gauss):
        box = self.get_leaves_box()
        p0 = np.zeros(self.nbin)
        for i_K in range(self.nbin):
            lower = np.squeeze(box[0, :, i_K])
            upper = np.squeeze(box[1, :, i_K])
            value, _ = mvn.mvnun(lower, upper, gauss[0], gauss[1])
            p0[i_K] = value
        return p0


class QuantTreeUnivariatePartitioning(QuantTreePartitioning):

    def __init__(self, pi_values):
        super().__init__(pi_values)

    def _build_partitioning(self, data):
        data = np.array(data).squeeze()

        ndata = len(data)
        self.ndata_training = ndata

        # Old version (compactly)
        # L = np.cumsum(np.round(self.pi_values * ndata)).astype(int) - 1

        # Updated version
        N = self.pi_values * ndata
        L = np.floor(N).astype(int)
        R = ndata - L.sum()
        if R > 0:
            L[np.argsort(N - L)[-R:]] += 1
        L = np.cumsum(L) - 1

        x = np.sort(data)

        self.leaves = np.concatenate(([-np.inf], x[L[:-1]], [np.inf]))

    def _find_bin(self, data):
        data = np.array(data).squeeze()
        if len(data.shape) == 0:
            data = [data]
        data = data.reshape(-1, 1)
        leaf = np.sum(data > self.leaves, axis=1) - 1

        return leaf


class QuantTreeThresholdStrategy(ThresholdStrategy):
    def __init__(self, nu: int = 0, ndata_training: int = 0, pi_values: Union[int, np.ndarray] = 2,
                 nbatch=100000, statistic_name: str = 'pearson'):
        super().__init__()
        self.N = ndata_training
        self.nu = nu
        self.statistic_name = statistic_name

        pi_values = np.array(pi_values)
        if pi_values.size == 1:
            self.K = pi_values
            self.pi_values = np.ones(self.K) / self.K
            self.is_unif = True
        else:
            self.pi_values = pi_values
            self.K = len(self.pi_values)
            if np.var(pi_values) < 1e-20:
                self.is_unif = True
            else:
                self.is_unif = False

        self.nbatch = nbatch
        self.thresholds_path = os.path.join(pkg_folder(), "thresholds", "quanttree_thresholds", "all_distr_quanttree.pkl")
        if not os.path.exists(self.thresholds_path):
            raise FileNotFoundError(f"Cannot find QuantTree thresholds file {self.thresholds_path}.")

    def get_threshold(self, alpha: float):
        if self.is_unif:
            with open(self.thresholds_path, 'rb') as f:
                all_thresholds = pickle.load(f)
            tkey = (self.statistic_name, self.K, self.N, self.nu)
            if tkey not in all_thresholds:
                print("[WARNING] Thresholds for this setting not found. Computing new thresholds...")
                warnings.warn("Contact diego.stucchi@polimi.it or giacomo.boracchi@polimi.it for an optimized version of the threshold computation.")
                return self.compute_threshold(alpha=alpha)
            else:
                thresholds, ecdf = all_thresholds[tkey]
                return thresholds[np.sum(ecdf <= 1 - alpha)]
        else:
            warnings.warn(
                "Thresholds are pre-computed only for uniform bin probabilities. See README.md at https://github.com/diegocarrera89/quantTree for more info.")
            warnings.warn(
                "Contact diego.stucchi@polimi.it or giacomo.boracchi@polimi.it for an optimized version of the threshold computation.")
            return self.compute_threshold(alpha=alpha)

    def add_threshold(self):
        raise NotImplementedError("Threshold computation not implemented. See README.md at https://github.com/diegocarrera89/quantTree for more info.")

    def estimate_quanttree_sim(self):
        partitioning = QuantTreeUnivariatePartitioning(self.pi_values)
        y = self.pi_values * self.nu

        stats = np.zeros(self.nbatch)
        for i_batch in range(self.nbatch):
            data = np.random.uniform(0, 1, self.N)
            batch = np.random.uniform(0, 1, self.nu)

            partitioning.build_partitioning(data)
            y_hat = partitioning.get_bin_counts(batch)

            if self.statistic_name == 'pearson':
                stats[i_batch] = np.sum(np.abs(y - y_hat) ** 2 / y)
            elif self.statistic_name == 'tv':
                stats[i_batch] = 0.5 * np.sum(np.abs(y - y_hat))
            else:
                ValueError('Statistic not supported')

        return stats

    def compute_threshold(self, alpha: float):
        stats = self.estimate_quanttree_sim()
        thresholds = np.unique(stats)
        ecdf = np.array([np.sum(stats <= cstat) / stats.shape[0] for cstat in thresholds])
        return thresholds[np.sum(ecdf <= 1-alpha)]


class QuantTree(Histogram):
    """
    Implementation of the QuantTree change detection method

    ...

    Attributes
    ----------
    partitioning : QuantTreePartitioning
        underlying input space partitioning by means of the QuantTree algorithm
    statistic : Statistic
        statistic object
    statistic_name : str
        string representing the adopted statistic
    pi_values : np.ndarray
        bin probabilities
    nu : int
        number of samples per batch
    alpha : float
        target False Positive Rate (FPR)
    threshold_strat : ThresholdStrategy
        object handling thresholds (depends on threshold_method)
    ndata_training : int
        number of training points (computed on train)
    threshold : float
        detection threshold (computed on train)
    """

    def __init__(self, pi_values, transformation_type, statistic_name, nu, alpha):
        """
        Parameters
        ----------
        pi_values : int or sequence of float
            If pi_values is int, it is the number of bins in the histogram, to be constructed with uniform probabilities.
            If pi_values is a sequence of float, it is the probability of each histogram bin, and the number of bins is
            simply the length of the sequence

        transformation_type :  {"none", "pca"}
            Transformation to be applied as preprocessing to the data.

        statistic_name : {"pearson", "tv"}
            Statistic employed by the method.

        nu : int
            Number of samples per batch in the considered batch-wise monitoring.

        alpha : float
            Desired percentage of false positives. The computation of the detection threshold depends on this value.
        """
        qtree = QuantTreePartitioning(pi_values=pi_values, transformation_type=transformation_type)
        super().__init__(partitioning=qtree, statistic_name=statistic_name)

        # qtree = QuantTreePartitioning(pi_values=pi_values, transformation_type=transformation_type)
        self.nu = nu
        self.alpha = alpha

        # More data that will be updated at training time
        self.ndata_training = None
        self.threshold = None
        self.threshold_strat = None

    def train_model(self, data, do_thresholds=True):
        # data.shape == (ndata_training, dim)
        assert (len(data.shape) == 2)

        self.partitioning.build_partitioning(data)  # Build the QuantTree histogram
        self.estimate_probabilities(data)  # Compute the actual pi_values
        self.set_statistic()  # Set the statistic function according to pi_values

        # Compute the threshold
        self.ndata_training = data.shape[0]

        if do_thresholds:
            self.compute_threshold()

    def compute_threshold(self):
        self.threshold_strat = QuantTreeThresholdStrategy(nu=self.nu,
                                                          ndata_training=self.ndata_training,
                                                          pi_values=self.pi_values,
                                                          statistic_name=self.statistic_name)

        self.threshold = self.threshold_strat.get_threshold(alpha=self.alpha)
