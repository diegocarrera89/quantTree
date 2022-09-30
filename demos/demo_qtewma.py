from quanttree import QT_EWMA, QuantTree
from quanttree.baselines import OnlineQuantTree, OnlineSPLL
from quanttree import utils
import numpy as np

"""
In this demo, we demonstrate the usage of QT-EWMA [1] and compare it against QuantTree [2] and SPLL [3].

In a concept drift experiment, we train each method over a training set of `N` samples drawn from the stationary
distribution. Then, we monitor a stream of samples initially drawn from the stationary distribution but containing a
change point `tau`, after which data are samples from the post-change distribution.

The stationary distribution is a 0-mean Gaussian distribution with a random covariance matrix.
The post-change distribution is a Gaussian distribution with fixed Kullback-Leibler distance to the stationary one,
generated using the Controlled Change Magnitude (CCM) framework [4].

See `demo_quanttree` for more information about training and monitoring with QuantTree.

QT-EWMA computes time-dependent detection thresholds to achieve a desired Average Run Length (ARL_0), namely, the number
of stationary sampled processed before raising a false alarm.

We run `nexp` concept drift detection experiments and report:
- the Average Run Length (ARL_0), together with the target set before the experiment;
- the average detection delay, i.e., the number of samples processed after the change point but before detecting the drift.

Parameters
----------
n_exp : int
    The number of concept drift detection experiments to perform
dim : int
    The data dimension
change_sKL : float
    The symmetric Kullback-Leibler distance between the stationary and post-change distributions
N : int
    The number of training samples drawn from the stationary distribution
K : int
    The number of bins constructed by the QuantTree algorithm
lam : float
    The weight assigned by QT-EWMA to the incoming samples.
nu : int
    The number of samples per batch in the considered batch-wise scenario (only for QuantTree and SPLL)
ngauss : int
    The number of Gaussian components fitted by the GMM (only for SPLL)

l_sequence : int
    The number of samples in the stream to be monitored
cp : int
    The index of the change point
ARL_0 : int
    The target Average Run Length (ARL_0), namely, the number of stationary samples monitored before a false alarm
target_fa : float
    The target percentage of false alarms (depends on ARL_0)

References
----------
[1] "Change Detection in Multivariate Datastreams Controlling False Alarms"
L. Frittoli, D. Carrera, G. Boracchi, Joint European Conference on Machine Learning and Knowledge Discovery in Databases (ECML-PKDD), 2021.

[2] "QuantTree: Histograms for Change Detection in Multivariate Data Streams"  
G. Boracchi, D. Carrera, C. Cervellera, D. Macciò, International Conference on Machine Learning (ICML) 2018.

[3] "Change Detection in Streaming Multivariate Data Using Likelihood Detectors"
L. Kuncheva, IEEE Transactions on Knowledge and Data Engineering (TKDE) 2013.

[4] "CCM: Controlling the change magnitude in high dimensional data"
C. Alippi, G. Boracchi, D. Carrera, INNS Conference on Big Data, 2016
"""


if __name__ == '__main__':
    methods = ["QT-EWMA", "QT", "SPLL"]

    # --- Demo parameters
    n_exp = 100                     # number of iterations of the detection experiment

    dim = 2                         # data dimension
    change_sKL = 1                  # symmetric Kullback-Leibler distance between the distributions
    N = 4096                        # number of training points
    K = 32                          # number of bins of the QuantTree histogram
    lam = 0.03                      # lambda parameter of QT-EWMA
    nu = 32                         # number of samples per batch (only for QuantTree)
    ngauss = 1                      # number of Gaussian components (only for SPLL)

    l_sequence = 10000              # lenght of the sequence to be monitored
    cp = 300                        # change point
    ARL_0 = 1000                    # target ARL_0, to be chosen in [500, 1000, 2000, 5000, 10000, 20000]
    target_fa = 1-(1-1/ARL_0)**cp   # target FPR (computed from the ARL_0)

    detection_times = {method: np.zeros(n_exp) for method in methods}  # to compute the ARL_0
    stopping_times = {method: np.zeros(n_exp) for method in methods}  # to compute the detection delay

    for j in range(n_exp):
        # --- Generating the distributions

        # Generate a random gaussian distribution
        gauss0 = utils.random_gaussian(dim)

        # Generate a random roto-translation yielding a changed desitribution with the desired Kullback-Leibler divergence
        rot, shift = utils.compute_roto_translation(gauss0, change_sKL)

        # Compute the alternative distribution
        gauss1 = utils.rotate_and_shift_gaussian(gauss0, rot, shift)

        # --- Generating training data and datastreams

        # Generate stationary data
        tr_data = np.random.multivariate_normal(gauss0[0], gauss0[1], N)

        # Sequence with no change (to computate the ARL_0)
        sequence0 = np.random.multivariate_normal(gauss0[0], gauss0[1], 6 * ARL_0)

        # Sequence with change (to compute the detection delay)
        pre = np.random.multivariate_normal(gauss0[0], gauss0[1], cp)
        post = np.random.multivariate_normal(gauss1[0], gauss1[1], l_sequence - cp)
        sequence1 = np.concatenate((pre, post))

        # --- Training and monitoring

        # QT-EWMA
        qtewma = QT_EWMA(pi_values=K, transformation_type='none', ARL_0=ARL_0, lam=lam)
        # Training
        qtewma.train_model(tr_data)
        # Monitoring
        stopping_times['QT-EWMA'][j] = qtewma.monitor(sequence0)
        detection_times['QT-EWMA'][j] = qtewma.monitor(sequence1)

        # QuantTree
        qtree = QuantTree(pi_values=K, transformation_type='none', statistic_name='pearson', nu=nu, alpha=target_fa)
        ol_qtree = OnlineQuantTree(ARL_0=ARL_0, qtree=qtree)
        # Training
        ol_qtree.train_model(tr_data)
        # Monitoring
        stopping_times['QT'][j] = ol_qtree.monitor(sequence0)
        detection_times['QT'][j] = ol_qtree.monitor(sequence1)

        # SPLL
        spll = OnlineSPLL(ngauss=ngauss, nu=nu, ARL_0=ARL_0)
        # Training
        spll.train_model(train_data=tr_data)
        # Monitoring
        stopping_times['SPLL'][j] = spll.monitor(sequence0)
        detection_times['SPLL'][j] = spll.monitor(sequence1)

    print(f" --- Results averaged over {n_exp} experiments ---")
    print(f"{'method':15.15s} {'delay':8.8s} {'FA rate':8.8s} {'(target)':8.8s} {'ARL_0':8.8s} {'(target)':8.8s}")
    for method in methods:
        tp = np.where(detection_times[method] >= cp)
        fp = np.where((detection_times[method] < cp) & (detection_times[method] > -1))
        avg_detection_delay = np.mean(detection_times[method][tp] - cp)
        fa_rate = len(fp[0]) / n_exp
        empirical_ARL = np.mean(stopping_times[method])
        print(
            f"{method:15.15s} {avg_detection_delay:8.2f} {fa_rate:8.2f} {target_fa:8.2f} {empirical_ARL:8.2f} {ARL_0:8.2f}")
