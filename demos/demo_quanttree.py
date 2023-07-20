import numpy as np
from quanttree import QuantTree
from quanttree import utils

"""
In this demo, we demonstrate the usage of QuantTree [1] and illustrate the parameters that can be customized.

The stationary distribution is a 0-mean Gaussian distribution with a random covariance matrix.
The post-change distribution is a Gaussian distribution with fixed Kullback-Leibler distance to the stationary one,
generated using the Controlled Change Magnitude (CCM) framework [2].

The QuantTree histogram is constructed over `N` samples drawn from the stationary distribution.
During training, QuantTree computes a threshold set to achieve a False Positive Rate (FPR) of `alpha`.
After training, we generate `nbatches` stationary and `nbatches` post-change batches and compute the QuantTree test
statistic of choice. We detect those batches whose statistic exceeds the detection threshold computed during training.

We report the FPR and TPR achieved by the QuantTree.

Parameters
----------
dim : int
    The data dimension
K : int
    The number of bins constructed by the QuantTree algorithm
N : int
    The number of training samples drawn from the stationary distribution
nu : int
    The number of samples per batch in the considered batch-wise scenario
alpha : float
    The desired percentage of false positives
statistic_name : {"pearson", "tv"}
    The statistic of choice (Pearson or Total Variation)
threshold_method : {"quanttree", "dirichlet"}
    The employed threshold computation strategy.
        "quanttree" uses the Monte Carlo simulations illustrated in the original paper [1]
        "dirichlet" uses the efficient strategy proposed in [3]
target_sKL : float
    The symmetric Kullback-Leibler distance between the stationary and post-change distributions
nbatches : int
    The number of stationary/post-change batches generated for testing
transf_type : {"none", "pca"}
    The transformation applied to preprocess the data.
        "none" means that no preprocessing is adopted
        "pca" menas that a PCA-transformation is fitted to the training data and used as preprocessing

References
----------
[1] "QuantTree: Histograms for Change Detection in Multivariate Data Streams"  
G. Boracchi, D. Carrera, C. Cervellera, D. Macciò, International Conference on Machine Learning (ICML) 2018.

[2] "CCM: Controlling the change magnitude in high dimensional data"
C. Alippi, G. Boracchi, D. Carrera, INNS Conference on Big Data, 2016

[3] "Nonparametric and Online Change Detection in Multivariate Datastreams using QuantTree"  
L. Frittoli, D. Carrera, G. Boracchi, IEEE Transactions on Knowledge and Data Engineering (TKDE) 2022.
"""

if __name__ == '__main__':
    # --- Demo parameters
    dim = 8                         # data dimension
    K = 32                          # number of bins of the histogram
    N = 4096                        # number of samples in the training set
    nu = 64                         # number of samples in each batch
    alpha = 0.05                    # target False Positive Rate
    statistic_name = 'pearson'      # chosen statistic ('pearson' or 'tv')

    target_sKL = 1                  # symmetric Kullback-Leibler distance between the distributions
    nbatches = 5000                 # Number of batches for testing
    transf_type = 'pca'             # 'none', 'pca'

    print(f"Data dimension:             {dim}")
    print(f"Number of training points:  {N}")
    print(f"Number of bins:             {K}")
    print(f"Number of points per batch: {nu}")
    print(f"Employed statistic:         {statistic_name}")

    # --- Generating the distributions

    # Generate a random gaussian distribution
    gauss0 = utils.random_gaussian(dim)

    # Generate a random roto-translation yielding a changed desitribution with the desired Kullback-Leibler divergence
    rot, shift = utils.compute_roto_translation(gauss0, target_sKL)

    # Compute the alternative distribution
    gauss1 = utils.rotate_and_shift_gaussian(gauss0, rot, shift)

    # --- Generating training data

    # Generate stationary data
    data = np.random.multivariate_normal(gauss0[0], gauss0[1], N)

    # --- Building and training a QuantTree model

    # QuantTree creation (automatically creates a Partitioning and initializes a ThresholdStrategy)
    qtree = QuantTree(pi_values=K, transformation_type=transf_type, statistic_name=statistic_name, nu=nu, alpha=alpha)

    # The QuantTree training procedure consists in:
    # i.   builds the QuantTree histogram
    # ii.  computes the actual pi_values according to training data
    # iii. computes the threshold as described in the paper
    qtree.train_model(data=data)

    # --- Testing
    print(f"\nTarget FPR = {alpha} ({alpha * 100}%)")
    print(f"Threshold = {qtree.threshold}\n")

    # Data ~ normal distribution
    test_batches_normal = np.random.multivariate_normal(gauss0[0], gauss0[1], (nbatches, nu))
    print(f"Stationary batches: {test_batches_normal.shape}")
    print("Computing statistic...")
    normal_stats = qtree.assess_goodness_of_fit(test_batches_normal)
    print(f"FPR = {100 * np.sum(normal_stats > qtree.threshold) / nbatches}%\n")

    # Data ~ change distribution
    test_batches_change = np.random.multivariate_normal(gauss1[0], gauss1[1], (nbatches, nu))
    print(f"Change batches: {test_batches_change.shape}")
    print("Computing statistic...")
    change_stats = qtree.assess_goodness_of_fit(test_batches_change)
    print(f"TPR: {100 * np.sum(change_stats > qtree.threshold) / nbatches}%")
