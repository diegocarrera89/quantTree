import numpy as np

from quanttree.cdm_src import CDM_QT_EWMA
from quanttree.qtewma_src import QT_EWMA

VISUALIZE_RESULTS = False  # requires matplotlib

"""
In this demo, we demonstrate the usage of CDM [1] individually monitoring two classes in a stream of data, and we
compare it against QT-EWMA [2], which monitors the distribution of the whole datastream.

The stationary distribution consists of two 1-dimensional uniform distributions, namely, U(0,1) and U(-0.5, 0.5).
After the change point `tau`, only the first class distribution drifts and becomes U(-0.5, 0.5). 

We plot the test statistic and the detection thresholds of the considered methods.

Parameters
----------
seed : int
    A seed to be fed to numpy for experimental reproducibility
training_points_per_class : int
    The number of training samples drawn from each stationary distribution
tau : int
    The index of the change point
points_after_tau : int
    The number of post-change samples generated after tau
ARL_0 : int
    The target Average Run Length (ARL_0), namely, the number of stationary samples monitored before a false alarm
lam : float
    The weight assigned by QT-EWMA to the incoming samples.
K : int
    The number of bins constructed by the QuantTree algorithm

References
----------
[1] "Class Distribution Monitoring for Concept Drift Detection"
D. Stucchi, L. Frittoli, G. Boracchi, International Joint Conference on Neural Networks (IJCNN), 2022.

[2] "Change Detection in Multivariate Datastreams Controlling False Alarms"
L. Frittoli, D. Carrera, G. Boracchi, Joint European Conference on Machine Learning and Knowledge Discovery in Databases (ECML-PKDD), 2021.
"""


def get_stationary(npointsperclass: int, do_shuffle: bool = True):
    _points = np.concatenate([
        # np.random.uniform(low=0.25, high=1, size=npointsperclass),
        np.random.uniform(low=0, high=1, size=npointsperclass),
        np.random.uniform(low=-.5, high=.5, size=npointsperclass),
    ])
    _labels = np.concatenate([np.zeros(npointsperclass), np.ones(npointsperclass)])

    if do_shuffle:
        idxs = np.arange(2 * npointsperclass)
        np.random.shuffle(idxs)
        _points = _points[idxs]
        _labels = _labels[idxs]

    return _points, _labels


def get_postchange(npointsperclass: int, do_shuffle: bool = True):
    _points = np.concatenate([
        # np.random.uniform(low=-1, high=-.25, size=npointsperclass),
        np.random.uniform(low=-.5, high=.5, size=npointsperclass),
        np.random.uniform(low=-.5, high=.5, size=npointsperclass),
    ])
    _labels = np.concatenate([np.zeros(npointsperclass), np.ones(npointsperclass)])

    if do_shuffle:
        idxs = np.arange(2 * npointsperclass)
        np.random.shuffle(idxs)
        _points = _points[idxs]
        _labels = _labels[idxs]

    return _points, _labels


if __name__ == '__main__':
    # --- Demo parameters
    seed = 2020                         # seed for experiment reproducibility
    if seed is not None:
        np.random.seed(seed)

    training_points_per_class = 512     # number of training points
    tau = 400                           # change point
    points_after_tau = 600              # length of the post-change datastream
    ARL_0 = 1000                        # target Average Run Length
    lam = 0.03                          # weight of incoming samples in QT-EWMA
    K = 32                              # number of histogram bins

    low0, high0 = 0, 1                  # Uniform distribution statistics for the first class
    low1, high1 = -.5, .5               # Uniform distribution statistics for the second class
    low2, high2 = -.5, .5               # Uniform distribution statistics for the first class after the change

    # --- Generating the labeled datastream

    # Generate training data
    training_data, training_labels = get_stationary(npointsperclass=training_points_per_class)

    # Generate stationary data
    stationary_data, stationary_labels = get_stationary(npointsperclass=tau//2)

    # Generate post-change data
    postchange_data, postchange_labels = get_postchange(npointsperclass=points_after_tau//2)

    # Concatenating in a datastream
    stream = np.concatenate([stationary_data, postchange_data])
    labels = np.concatenate([stationary_labels, postchange_labels])

    # --- Training and monitoring

    # QT-EWMA
    qtewma = QT_EWMA(pi_values=K, transformation_type='pca', ARL_0=ARL_0, lam=lam)
    # Training over the whole training set
    qtewma.train_model(data=training_data.reshape(-1, 1))
    # Class-agnostic monitoring
    qtewma_tau_hat = qtewma.monitor(stream=stream)
    qtewma_statistics = qtewma.compute_statistic(stream=stream)
    qtewma_thresholds = qtewma.get_thresholds(stream_length=stream.shape[0])

    # CDM (w/ QT-EWMA)
    cdm = CDM_QT_EWMA(nqtree=training_points_per_class, ARL_0=ARL_0, lam=lam, K=K)
    # Training (separately trains two QT-EWMA on the individual classes)
    cdm.train(train_points=training_data.reshape(-1, 1), train_labels=training_labels)
    # Independently monitors the streams
    cdm_tau_hat = cdm.monitor(stream=stream, labels=labels)
    cdm_statistics = cdm.compute_statistics(stream=stream, labels=labels)
    cdm_thresholds = cdm.get_thresholds(streams_length=[stream[labels == i].shape[0] for i in [0, 1]])
    cdm_changed_class = labels[cdm_tau_hat]

    # --- Analysis of the results
    if qtewma_tau_hat == -1:
        print("QT-EWMA did not detect any change")
    else:
        print(f"QT-EWMA detected a change at {qtewma_tau_hat}")
        if qtewma_tau_hat < tau:
            print(" False alarm!")

    if cdm_tau_hat == -1:
        print("CDM did not detect any change")
    else:
        if cdm_changed_class == 1:
            print(f"CDM detected a change at {cdm_tau_hat} over class {int(cdm_changed_class)}")
            print(f"Since {int(cdm_changed_class)} has not changed, it is a false alarm.")
        else:
            print(f"CDM detected a change at {cdm_tau_hat} over class {int(cdm_changed_class)}")
            if cdm_tau_hat < tau // 2:
                print(f"Since {cdm_tau_hat} < {tau//2}, it is a false alarm!")

    if VISUALIZE_RESULTS:
        import matplotlib.pyplot as plt
        # # # # # # # # # # # # # # # # #
        # FIGURE 1 - Data visualization #
        # # # # # # # # # # # # # # # # #
        color = {0: 'red', 1: 'blue'}

        plt.figure(figsize=(16, 6))
        plt.scatter(np.arange(stream.shape[0]), stream, c=[color[label] for label in labels], alpha=.5)
        plt.axvline(tau, label='Change point', color='grey', linestyle='dashed')
        plt.ylim([-1.01, 1.01])
        plt.xlim([-1, tau + points_after_tau + 1])
        plt.xlabel("Time")
        plt.legend()
        plt.show()
        plt.close()

        # # # # # # # # # # # # # # # # # # # # #
        # FIGURE 2 - Statistics and thresholds  #
        # # # # # # # # # # # # # # # # # # # # #
        plt.figure(figsize=(16, 9))
        MAX_VALUE = max([
            max(qtewma_statistics),
            max([max(cdm_statistics[i]['statistics']) for i in cdm_statistics.keys()])
        ])

        plt.subplot(3, 1, 1)
        X = np.arange(qtewma_statistics.shape[0])
        plt.plot(X, qtewma_statistics, color='purple')
        plt.plot(X, qtewma_thresholds, color='black')
        plt.ylim([0, MAX_VALUE])
        plt.axvline(qtewma_tau_hat, color='green')
        plt.axvline(tau, color='grey', linestyle='dashed')

        for i in [0, 1]:
            plt.subplot(3, 1, i+2)
            plt.plot(cdm_statistics[i]['indexes'], cdm_statistics[i]['statistics'], color=color[i])
            plt.plot(cdm_statistics[i]['indexes'], cdm_thresholds[i], color='black')
            plt.ylim([0, MAX_VALUE])
            plt.axvline(tau, color='grey', linestyle='dashed')
            if cdm_changed_class == i:
                plt.axvline(cdm_tau_hat)
        plt.show()
        plt.close()
