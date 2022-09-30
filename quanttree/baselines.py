import os
from .core import DataModel, ThresholdStrategy, reshape_data, pearson_statistic_
import numpy as np
import scipy.stats
from sklearn.mixture import GaussianMixture
from quanttree import QuantTree


class BootstrapThresholdStrategy(ThresholdStrategy):

    def __init__(self, model: DataModel, nu: int, nbatch: int, data: np.ndarray):
        super().__init__()
        self.model = model
        self.nu = nu
        self.nbatch = nbatch
        self.data = reshape_data(data)

    def _get_threshold(self, alpha):
        ndata = self.data.shape[0]

        stats = []
        for i_batch in range(self.nbatch):
            batch = self.data[np.random.choice(ndata, self.nu, replace=True), :]
            stats.append(self.model.assess_goodness_of_fit(batch))

        stats.sort()
        stats.insert(0, stats[0] - 1)
        threshold = stats[np.int(np.floor((1 - alpha) * self.nbatch))]
        return threshold

    # non necessario da implementare, ma solo se serve configurarla su un modello
    def configure_strategy(self, model: DataModel, data: np.ndarray):
        self.model = model
        self.data = reshape_data(data)


class TwoSampleHotellingModel(DataModel):
    """Modello che implementa il two sample hotelling test"""

    def __init__(self):
        self.mu: np.ndarray = None
        self.Sigma: np.ndarray = None
        self.ntrain_data: int = None

    def train_model(self, data: np.ndarray):
        data = reshape_data(data)
        self.mu = np.mean(data, 0)
        self.Sigma = np.cov(np.transpose(data))
        self.ntrain_data = data.shape[0]

    def assess_goodness_of_fit(self, data: np.ndarray):
        data = reshape_data(data)
        mu0 = self.mu
        Sigma0 = self.Sigma
        n0 = self.ntrain_data
        mu1 = np.mean(data, 0)
        Sigma1 = np.cov(np.transpose(data))
        n1 = data.shape[0]
        Sigma_pool = (n0 - 1) / (n0 + n1 - 2) * Sigma0 + (n1 - 1) / (n0 + n1 - 2) * Sigma1
        Sigma_pool = (1 / n0 + 1 / n1) * Sigma_pool
        t = np.dot(mu0 - mu1, np.linalg.solve(Sigma_pool, mu0 - mu1))
        return t


class TwoSampleHotellingThresholdStrategy(ThresholdStrategy):

    def __init__(self, nu: int, ntrain_data: int = None, dim: int = None):
        super().__init__()
        self.n0 = ntrain_data
        self.n1 = nu
        self.dim = dim

    def _get_threshold(self, alpha):
        dof1 = self.dim
        dof2 = self.n0 + self.n1 + - self.dim - 1
        factor = (self.n0 + self.n1 - 2) * self.dim / (self.n0 + self.n1 + - self.dim - 1)
        return factor * scipy.stats.f.ppf(1 - alpha, dof1, dof2)

    def configure_strategy(self, model: TwoSampleHotellingModel, data: np.ndarray):
        data = reshape_data(data)
        self.n0 = model.ntrain_data
        self.dim = data.shape[1]


class GaussianMixtureDataModel(DataModel):

    def __init__(self, ngauss):
        self.ngauss = ngauss
        self.model = GaussianMixture(ngauss, covariance_type='tied')
        self.cov_inv = None

    def train_model(self, data: np.ndarray):
        data = reshape_data(data)
        self.model.fit(data)
        self.cov_inv = np.linalg.inv(self.model.covariances_)

    def assess_goodness_of_fit(self, data: np.ndarray):
        data = reshape_data(data)

        all_loglike = np.zeros((data.shape[0], self.ngauss))
        for ii in range(data.shape[0]):
            v = data[ii][np.newaxis, :] - self.model.means_
            all_loglike[ii, :] = np.array([np.matmul(np.matmul(x, self.cov_inv), x.T) for x in v])

        all_loglike = np.min(all_loglike, axis=1)
        return np.sum(all_loglike) / data.shape[0]


class ParametricGaussianModel(GaussianMixtureDataModel):
    """Modello che implementa una gaussiana usando la log likelihood come statistica"""

    def __init__(self):
        super().__init__(1)
        self.model = GaussianMixture(1, covariance_type='full')


class PearsonAsymptoticThresholdStrategy(ThresholdStrategy):

    def __init__(self, nbin):
        super().__init__()
        self.nbin = nbin

    def _get_threshold(self, alpha):
        return scipy.stats.chi2.isf(alpha, self.nbin - 1)


class OnlineQuantTree:
    def __init__(self, ARL_0, qtree: QuantTree):
        self.qtree = qtree
        self.K = self.qtree.partitioning.nbin
        self.nu = self.qtree.nu
        self.ntrain = None
        self.ARL_0 = ARL_0

    def train_model(self, train_data):
        if self.qtree is not None:
            Warning("QuantTree is already trained. Overwriting.")
        self.qtree.train_model(train_data, do_thresholds=False)
        self.K = self.qtree.partitioning.nbin
        self.nu = self.qtree.nu
        self.ntrain = train_data.shape[0]

    def monitor(self, sequence):
        h = self.get_threshold()

        def pearson(pi_exp, pi_hat, nu):
            return nu * (np.sum(pi_hat**2 / pi_exp) - 1)
        npts = sequence.shape[0]
        bins = self.qtree.partitioning.find_bin(sequence)
        K = self.K

        # matrix of indicator vectors (Y(k,i) = 1 iff bin(x_i) == k)
        Y = np.zeros((K, npts), dtype=int)
        Y[bins, np.arange(npts)] = 1

        p = np.ones(K) / K      # target density distribution
        b = 0   # batch number
        t = 0   # sample number
        change_detected = False
        while (b + 1) * self.nu < npts:     # continue until no points are left
            t_prev = self.nu * b
            t = self.nu * (b + 1)
            Z = np.average(Y[:, t_prev:t], axis=1)
            stat = pearson(p, Z, self.nu)
            if stat > h:
                change_detected = True
                break
            b += 1

        if change_detected:
            return t
        else:
            return -1

    def get_threshold(self):
        fld, _ = os.path.split(__file__)
        filename = os.path.join(fld, 'thresholds', 'qt_batchwise_thresholds', f'qt_batchwise_stats_{self.ntrain}_{self.K}_{self.nu}.npz')
        lookup = np.load(filename)['arr_0']
        alpha = self.nu / self.ARL_0
        return np.percentile(lookup, (1 - alpha) * 100)


class OnlineSPLL:
    def __init__(self, ngauss, nu, ARL_0):
        self.ngauss = ngauss
        self.nu = nu
        self.ARL_0 = ARL_0
        self.ntrain = None
        self.nboots = None
        self.gmm = None
        self.threshold = None

    def train_model(self, train_data):
        # train a GMM
        self.ntrain = train_data.shape[0] // 4
        self.nboots = train_data.shape[0] - self.ntrain
        self.gmm = GaussianMixtureDataModel(self.ngauss)
        self.gmm.train_model(train_data[:self.ntrain])
        self.get_threshold(train_data[self.ntrain:])

    def monitor(self, sequence):
        npts = sequence.shape[0]
        b = 0
        t = 0
        change_detected = False
        while (b + 1) * self.nu < npts:
            t0 = self.nu * b
            t = self.nu * (b + 1)
            stat = self.gmm.assess_goodness_of_fit(sequence[t0:t])
            if stat > self.threshold:
                change_detected = True
                break
            b += 1
        if change_detected:
            return t
        else:
            return -1

    def get_threshold(self, bootstrap_data):
        # bootstrap to compute threshold
        b = 0   # batch number
        H = []  # statistics
        while (b + 1) * self.nu < self.nboots:
            t0 = self.nu * b
            t1 = self.nu * (b + 1)
            H.append(self.gmm.assess_goodness_of_fit(bootstrap_data[t0:t1]))
            b += 1
        alpha = self.nu / self.ARL_0
        self.threshold = np.percentile(np.array(H), (1 - alpha) * 100)
