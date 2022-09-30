from quanttree.quanttree_src import QuantTreePartitioning
from quanttree.utils import pkg_folder
import numpy as np
import os


def get_target_probabilities(ntrain, nbins):
    pi_hats = (ntrain / (ntrain + 1)) * np.ones(nbins) / nbins
    pi_hats[-1] = ((ntrain / nbins) + 1) / (ntrain + 1)
    # pi_hats = np.ones(nbins) / nbins
    return pi_hats


def get_pwp(checkpoints, coeffs):
    def piecewise(x):
        pt = np.poly1d(coeffs[-1])
        y = pt(1 / x)
        for i, (t, c) in enumerate(zip(checkpoints[::-1], coeffs[-2::-1])):
            loc = x < t
            pt = np.poly1d(c)
            y[loc] = pt(1 / x[loc])
        return y

    return piecewise


class QT_EWMA:
    """
    Implementation of the QT-EWMA sequential change detection method

    ...

    Attributes
    ----------
    ARL_0 : int
        target Average Run Length
    lam : float
        weight associated to incoming samples
    threshold_mode : {"polynomial", "mixed"}
        thresholds approximation mode (see __init__)
    do_update : bool
        flag enabling the update of the bin probabilities during monitoring
    beta : float
        updating speed
    min_tstar : int
        minimum number of samples before detecting a change
    qtree : QuantTreePartitioning
        underlying input space partitioning by means of the QuantTree algorithm
    K : int
        number of bins in the histogram
    ntrain : int
        number of training samples
    thresholds : sequence of floats
        time-dependent thresholds computed as in []
    """

    def __init__(self, pi_values, transformation_type, ARL_0, lam,
                 threshold_mode='mixed', do_update=False, beta=None, min_tstar=5):
        """
        Parameters
        ----------
        pi_values : int or sequence of float
            If pi_values is int, it is the number of bins in the histogram, to be constructed with uniform probabilities.
            If pi_values is a sequence of float, it is the probability of each histogram bin, and the number of bins is
            simply the length of the sequence

        transformation_type :  {"none", "pca"}
            Transformation to be applied as preprocessing to the data.

        ARL_0 : int
            Target Average Run Length, namely, the desired number of stationary samples processed before raising a false alarm.

        lam : float
            Weight associated with the incoming samples in the EWMA statistic.

        threshold_mode : {"polynomial", "mixed"}
            Choice of the threshold approximation employed after the Monte Carlo simulation.
            "polynomial" uses a single polynomial in 1/t to approximate the empirical thresholds for every time t.
            "mixed" uses a piecewise polynomial

        do_update : bool
            If True, the bin probabilities are updated using the incoming samples.
            If False, the bin probabilities are fixed after training.

        beta : float
            Updating speed, i.e., the weights associated to new samples when updating the reference bin probabilities.

        min_tstar : int
            Minimum number of samples to be processed before raising a detection. The QT-EWMA statistic benefits from
            processing some samples before actually raising an alarm.
        """
        # QT-EWMA parameters
        self.ARL_0 = ARL_0
        self.lam = lam
        self.threshold_mode = threshold_mode
        self.do_update = do_update
        self.beta = beta
        self.min_tstar = min_tstar

        # QuantTree setup
        self.qtree = QuantTreePartitioning(pi_values=pi_values, transformation_type=transformation_type)
        self.K = self.qtree.nbin
        self.ntrain = -1
        self.thresholds = None

    def train_model(self, data):
        # data: (npts, dim)
        self.ntrain = data.shape[0]
        self.qtree.build_partitioning(data=data)

    def monitor(self, stream):
        # stream: (npts, dim)
        self.thresholds = self.get_thresholds(stream_length=stream.shape[0])

        # initialization
        z = get_target_probabilities(self.ntrain, self.K)
        pi = get_target_probabilities(self.ntrain, self.K)
        change_detected = False
        tstar = 0

        bins = self.qtree.find_bin(stream)
        for tstar, bin_tstar in enumerate(bins):
            x = np.zeros(self.K)
            x[bin_tstar] = 1
            z = (1 - self.lam) * z + self.lam * x

            if self.do_update:
                w_t = 1 / (self.beta * (self.ntrain + tstar + 1))
                pi = (1 - w_t) * pi + w_t * pi

            stat = np.sum((z - pi) ** 2 / pi)
            if stat > self.thresholds[tstar] and tstar >= self.min_tstar:
                change_detected = True
                break

        if change_detected:
            return tstar
        else:
            return -1

    def get_thresholds(self, stream_length):
        if self.do_update:
            raise NotImplementedError("Update mode has not been (re)implemented yet!")
        else:
            assert self.threshold_mode in ['polynomial', 'mixed']
            if self.threshold_mode == 'polynomial':
                polynomial_path = os.path.join(pkg_folder(), 'thresholds', 'qtewma_thresholds', 'polynomial',
                                               f'polynomial_qt_ewma_{self.ntrain}_{self.ARL_0}_{self.K}.npy')
                if not os.path.isfile(polynomial_path):
                    raise Exception(f"Cannot find threshold file: {polynomial_path}")
                polynomial_coeffs = np.load(polynomial_path)
                polynomial = np.poly1d(polynomial_coeffs)
                t = np.arange(1, stream_length + 1)
                return polynomial(1 / t)

            elif self.threshold_mode == 'mixed':
                mixed_path = os.path.join(pkg_folder(), 'thresholds', 'qtewma_thresholds', 'mixed',
                                          f'mixed_qt_ewma_{self.ntrain}_{self.ARL_0}_{self.K}.npz')
                if not os.path.isfile(mixed_path):
                    raise Exception(f"Cannot find threshold file: {mixed_path}")

                mixed_data = np.load(mixed_path)
                coeffs = mixed_data['coeffs']
                checkpoints = mixed_data['checkpoints']
                t = np.arange(1, stream_length + 1)
                piecewise = get_pwp(checkpoints, coeffs)
                return piecewise(t)

    def compute_statistic(self, stream):
        # initialization
        z = get_target_probabilities(self.ntrain, self.K)
        pi = get_target_probabilities(self.ntrain, self.K)
        statistic_stream = np.zeros(stream.shape[0])
        bins = self.qtree.find_bin(stream)
        for t, bin_tstar in enumerate(bins):
            x = np.zeros(self.K)
            x[bin_tstar] = 1
            z = (1 - self.lam) * z + self.lam * x
            statistic_stream[t] = np.sum((z - pi) ** 2 / pi)
        return statistic_stream
