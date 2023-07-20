import numpy as np

from quanttree import EuclideanKernelQuantTree, MahalanobisKernelQuantTree, WeightedMahalanobisKernelQuantTree
from quanttree import utils

if __name__ == '__main__':
    # --- Demo parameters
    dim = 8                         # data dimension
    K = 64                          # number of bins of the histogram
    N = 4096                        # number of samples in the training set
    nu = 128                        # number of samples in each batch
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

    # Data ~ normal distribution
    test_batches_normal = np.random.multivariate_normal(gauss0[0], gauss0[1], (nbatches, nu))
    print(f"Stationary batches: {test_batches_normal.shape}")

    # Data ~ post-change distribution
    test_batches_change = np.random.multivariate_normal(gauss1[0], gauss1[1], (nbatches, nu))
    print(f"Post-change batches: {test_batches_change.shape}")

    # --- Building and training the Kernel QuantTree models

    kqts = {}

    kqts['euclidean'] = EuclideanKernelQuantTree(
        pi_values=K, nu=nu, alpha=alpha, statistic_name=statistic_name, transformation_type='none', nbatch=100000,
        centroid_selection_metric='information_gain',
        centroid_selection_criterion='max', centroid_selection_picks=250
    )

    kqts['mahalanobis'] = MahalanobisKernelQuantTree(
        pi_values=K, nu=nu, alpha=alpha, statistic_name=statistic_name, transformation_type='none', nbatch=100000,
        centroid_selection_metric='information_gain',
        centroid_selection_criterion='max', centroid_selection_picks=250, use_global_covariance=True
    )

    kqts['weighted'] = WeightedMahalanobisKernelQuantTree(
        pi_values=K, nu=nu, alpha=alpha, statistic_name=statistic_name, transformation_type='none', nbatch=100000,
        centroid_selection_metric='information_gain',
        centroid_selection_criterion='max', centroid_selection_picks=250
    )

    models_list = ['euclidean', 'mahalanobis', 'weighted']

    # --- Testing
    print(f"\nTarget FPR = {alpha} ({alpha * 100}%)")

    for model_name in models_list:
        print(f"--- Model: {model_name} ---")
        kqts[model_name].train_model(training_data=data)

        print(f"Threshold = {kqts[model_name].threshold}")
        print("Computing statistics on stationary batches...")
        normal_stats = kqts[model_name].assess_goodness_of_fit(test_batches_normal)
        print(f"FPR = {100 * np.sum(normal_stats > kqts[model_name].threshold) / nbatches}%\n")

        print("Computing statistics on post-change batches...")
        change_stats = kqts[model_name].assess_goodness_of_fit(test_batches_change)
        print(f"TPR: {100 * np.sum(change_stats > kqts[model_name].threshold) / nbatches}%\n")
