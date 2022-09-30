from typing import List

import numpy as np
from quanttree import QT_EWMA


class CDM_QT_EWMA:
    """
    Implementation of the Class-Distribution Monitoring method employing several QT-EWMA instances as detectors.

    ...

    Attributes
    ----------
    nqtree : int
        number of training points used to train the QT-EWMA instances
    ARL_0 : int
        target Average Run Length
    lam : float
        weight associated to incoming samples
    K : int
        number of bins in the histogram
    qt_ewma : list of QT_EWMA
        QT_EWMA instances used to monitor the individual classes
    labels_list : list
        labels represented in the training set
    transformation_type : {"none", "pca"}
        transformation used for data preprocessing
    threshold_mode : {"mixed", "polynomial"}
        See QT-EWMA
    min_tstar : int
        minimum number of samples before detecting a change
    nclasses: int
        number of classes to monitor
    """

    def __init__(self, nqtree, ARL_0, lam, K,
                 transformation_type='pca',
                 threshold_mode='mixed', min_tstar=5):
        """
        Parameters
        ----------
        nqtree : int
            The number of training points per class

        ARL_0 : int
            Target Average Run Length, namely, the desired number of stationary samples processed before raising a false alarm.

        lam : float
            Weight associated with the incoming samples in the EWMA statistic.

        K : int
            Number of bins in the histograms

        transformation_type :  {"none", "pca"}
            Transformation to be applied as preprocessing to the data.

        threshold_mode : {"polynomial", "mixed"}
            Choice of the threshold approximation employed after the Monte Carlo simulation.
            "polynomial" uses a single polynomial in 1/t to approximate the empirical thresholds for every time t.
            "mixed" uses a piecewise polynomial

        min_tstar : int
            Minimum number of samples to be processed before raising a detection. The QT-EWMA statistic benefits from
            processing some samples before actually raising an alarm.
        """
        self.nqtree = nqtree
        self.ARL_0 = ARL_0
        self.lam = lam
        self.K = K
        self.qt_ewma: List[QT_EWMA] = []
        self.labels_list = None
        self.transformation_type = transformation_type
        self.threshold_mode = threshold_mode
        self.min_tstar = min_tstar
        self.nclasses = None

    def train(self, train_points, train_labels):
        self.labels_list = np.unique(train_labels)
        self.nclasses = len(self.labels_list)

        # Split the stream into class-specific substreams
        train_stream = {i: train_points[train_labels == i] for i in self.labels_list}
        assert all([train_stream[i].shape[0] >= self.nqtree for i in self.labels_list]), \
            "[QT_EWMA] Not enough points to train a QuantTree"

        # Build individual QuantTrees over the substreams
        self.qt_ewma = {}
        for i in self.labels_list:
            self.qt_ewma[i] = QT_EWMA(pi_values=self.K, transformation_type=self.transformation_type,
                                      ARL_0=self.ARL_0, lam=self.lam, threshold_mode=self.threshold_mode,
                                      min_tstar=self.min_tstar)
            self.qt_ewma[i].train_model(data=train_stream[i][:self.nqtree])

    def monitor(self, stream, labels):
        # Perform stream-wise monitoring
        tau_hats = []
        idxs = np.arange(stream.shape[0])
        for i in self.labels_list:
            loc_i = labels == i
            sub_tau_hat = self.qt_ewma[i].monitor(stream[loc_i])
            if sub_tau_hat > 0:
                tau_hats.append(idxs[loc_i][sub_tau_hat])

        if len(tau_hats) == 0:
            return -1
        else:
            return np.min(tau_hats).astype(int)

    def compute_statistics(self, stream, labels):
        statistic_streams = {i: {'statistics': None, 'indexes': None} for i in range(self.nclasses)}

        idxs = np.arange(stream.shape[0])
        for i in self.labels_list:
            loc_i = labels == i
            statistic_streams[i]['statistics'] = self.qt_ewma[i].compute_statistic(stream=stream[loc_i])
            statistic_streams[i]['indexes'] = idxs[loc_i]

        return statistic_streams

    def get_thresholds(self, streams_length):
        return {
            i: self.qt_ewma[i].get_thresholds(stream_length=stream_length)
            for i, stream_length in enumerate(streams_length)
        }
