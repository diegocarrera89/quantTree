import numpy as np
from typing import Union, Callable
from scipy.spatial.distance import pdist, cdist
from quanttree.kqt_base import NodeFunction
from scipy.spatial import ConvexHull


def entropy(points: np.ndarray):
    # points: (npoints, dim)
    covariance = np.cov(points, rowvar=False)
    det_cov = np.linalg.det(covariance)
    return .5 * np.log(det_cov)


def entropy_complete(points: np.ndarray):
    # points: (npoints, dim)
    covariance = np.cov(points, rowvar=False)
    det_cov = np.linalg.det(covariance)
    dim = points.shape[1]
    constant = .5 * (np.log(2 * np.pi) + 1)
    return dim * constant + .5 * np.log(det_cov)


def information_gain(points: np.ndarray, in_bin: np.ndarray):
    return fast_information_gain(points=points, in_bin=in_bin)


def slow_information_gain(points: np.ndarray, in_bin: np.ndarray):
    npoints = points.shape[0]
    npoints_in_bin = np.sum(in_bin)
    npoints_not_in_bin = npoints - npoints_in_bin

    in_factor = npoints_in_bin / npoints
    not_in_factor = npoints_not_in_bin / npoints

    return entropy(points) - in_factor * entropy(points[in_bin]) - not_in_factor * entropy(points[~in_bin])


def multi_variance(data: np.ndarray):
    return np.linalg.det(np.cov(data, rowvar=False))


def maximal_multidistance(data: np.ndarray):
    return np.max(pdist(data))


def smallest_ball_radius(data: np.ndarray, center: np.ndarray):
    return np.min(np.max(cdist(data, center.reshape((1, -1)))))


def mahalanobis_multidistance(data: np.ndarray):
    invcovmat = np.linalg.inv(np.cov(data, rowvar=False))
    x_mean = data.mean(axis=0)
    centered_data = data - x_mean
    return np.sum(np.matmul(centered_data, invcovmat) * centered_data, axis=1) ** .5


def get_score_increase(data: np.ndarray,
                       score_function: Callable,
                       in_bin: np.ndarray):
    assert score_function in [entropy, multi_variance, maximal_multidistance, mahalanobis_multidistance]

    global_score = score_function(data)
    in_score = score_function(data[in_bin])
    out_score = score_function(data[~in_bin])

    npoints = data.shape[0]
    in_factor = np.sum(in_bin).astype(int) / npoints
    out_factor = 1 - in_factor

    return global_score - in_factor * in_score - out_factor * out_score


def get_score_function(function_name: str):
    if function_name not in ["entropy", "multi_variance", "mahalanobis_multidistance", "maximal_multidistance"]:
        raise ValueError("Invalid score function name")
    if function_name == 'entropy':
        return entropy
    if function_name == 'multi_variance':
        return multi_variance
    if function_name == 'maximal_multidistance':
        return maximal_multidistance
    if function_name == 'mahalanobis_multidistance':
        return mahalanobis_multidistance


def find_centroid(data: np.ndarray, npartition: int,
                  kernel: NodeFunction,
                  direction: str,
                  score_function: Union[Callable, str],
                  nreps: int = 250,
                  get_scores: bool = False):
    assert direction in ['low', 'high']

    all_scores = []
    if type(score_function) == str:
        score_function = get_score_function(score_function)

    nsamples, _ = data.shape
    nreps = min(nreps, nsamples)

    if nsamples == nreps:
        indexes = np.arange(nsamples)
    else:
        indexes = np.random.choice(nsamples, nreps, replace=False)

    best_score_increase: float = -1.0
    best_index: int = -1

    for index in indexes:
        current_centroid = data[index]

        kernel.centroid = current_centroid
        distances = kernel(data)

        sorted_distances = np.sort(distances)
        if direction == 'low':
            current_threshold = sorted_distances[npartition - 1]
            in_bin = distances <= current_threshold
        elif direction == 'high':
            current_threshold = sorted_distances[-npartition]
            in_bin = distances >= current_threshold
        else:
            raise ValueError("Invalid direction")

        in_bin = np.array(in_bin)
        current_score_increase = get_score_increase(data, score_function, in_bin)
        all_scores.append(current_score_increase)

        if current_score_increase > best_score_increase:
            best_score_increase = current_score_increase
            best_index = index

    if not get_scores:
        return data[best_index], best_score_increase
    else:
        return data[best_index], best_score_increase, np.array(all_scores)[indexes.argsort()]


def absolute_density(radius, npoints):
    # TODO: elevate to dimension?
    return npoints / radius


def maximize_absolute_density(
        data: np.ndarray,
        potential_centroids: np.ndarray,
        kernel: NodeFunction,
        direction: str,
        npartition: int,
        nreps: int = -1):
    nsamples, _ = data.shape
    ncentroids, _ = potential_centroids.shape

    nreps = min(nreps, ncentroids)
    if nreps == -1:
        indexes = np.arange(ncentroids)
    else:
        indexes = np.random.choice(nsamples, nreps, replace=False)

    best_metric: float = -1.0
    best_index: int = -1

    all_metrics = []
    for index in indexes:
        current_centroid = potential_centroids[index]
        kernel.centroid = current_centroid
        distances: np.ndarray = kernel(data=data)
        sorted_distances = np.sort(distances)

        radius = None
        npoints_in_bin = None
        if direction == 'low':
            current_threshold = sorted_distances[npartition - 1]
            npoints_in_bin = np.sum(kernel(potential_centroids) <= current_threshold)
            radius = current_threshold
        if direction == 'high':
            current_threshold = sorted_distances[-npartition]
            npoints_in_bin = np.sum(kernel(potential_centroids) >= current_threshold)
            radius = 1 / current_threshold
        current_metric = absolute_density(radius=radius, npoints=npoints_in_bin)
        all_metrics.append(current_metric)
        if current_metric > best_metric:
            best_metric = current_metric
            best_index = index

    return potential_centroids[best_index], best_metric


def fast_information_gain(points: np.ndarray, in_bin: np.ndarray):
    return - (np.sum(in_bin) * entropy(points[in_bin]) + (points.shape[0] - np.sum(in_bin)) * entropy(points[~in_bin]))


def minimize_convex_hull(
        data: np.ndarray,
        kernel: NodeFunction,
        direction: str,
        npartition: int,
        potential_centroids: np.ndarray = None,
        nreps: int = -1):
    if potential_centroids is None:
        potential_centroids = data

    nsamples, _ = data.shape
    ncentroids, _ = potential_centroids.shape

    nreps = min(nreps, ncentroids)
    if nreps == -1:
        indexes = np.arange(ncentroids)
    else:
        indexes = np.random.choice(ncentroids, nreps, replace=False)

    best_metric: float = np.inf
    best_index: int = -1

    all_metrics = []
    for index in indexes:
        current_centroid = potential_centroids[index]
        kernel.centroid = current_centroid
        distances: np.ndarray = kernel(data=data)
        sorted_distances = np.sort(distances)

        in_bin = None
        if direction == 'low':
            current_threshold = sorted_distances[npartition - 1]
            in_bin = np.array(kernel(data) <= current_threshold)
        if direction == 'high':
            current_threshold = sorted_distances[-npartition]
            in_bin = np.array(kernel(data) >= current_threshold)

        current_convex_hull = ConvexHull(points=data[in_bin]).volume
        all_metrics.append(current_convex_hull)
        if current_convex_hull < best_metric:
            best_metric = current_convex_hull
            best_index = index

    return potential_centroids[best_index], best_metric


def gini_paolo(arr):
    # first sort
    sorted_arr = arr.copy()
    sorted_arr.sort()
    n = arr.size
    coef_ = 2. / n
    const_ = (n + 1.) / n
    weighted_sum = sum([(i + 1) * yi for i, yi in enumerate(sorted_arr)])
    return coef_ * weighted_sum / (sorted_arr.sum()) - const_


def gini(values: np.ndarray):
    nvalues = len(values)
    sorted_values = np.sort(values)
    sorted_cumulatives = np.cumsum(sorted_values)
    total = sorted_cumulatives[-1]
    sorted_cumulatives = sorted_cumulatives / total
    return 1 - 2 * (np.sum(sorted_cumulatives) - 1 / 2) / nvalues


def optimize_metric(quality_metric: str, minmax: str, data: np.ndarray, kernel: NodeFunction, num_inbin: int,
                    npicks: int = 250, **kwargs):
    assert quality_metric in ['information_gain', 'gini']
    assert minmax in ['min', 'max']

    nsamples = data.shape[0]

    if npicks == -1 or npicks >= nsamples:
        npicks = nsamples
        indexes = np.arange(nsamples)
    else:
        indexes = np.random.choice(nsamples, size=npicks, replace=False)

    metrics = np.zeros(npicks)
    for i, idx in enumerate(indexes):
        centroid = data[idx]
        distances: np.ndarray = kernel(data, centroid=centroid)
        threshold = distances[np.argsort(distances)[num_inbin - 1]]

        if quality_metric == 'information_gain':
            metrics[i] = fast_information_gain(points=data, in_bin=distances <= threshold)
        if quality_metric == 'gini':
            metrics[i] = gini(values=distances)

    sorted_indexes = np.argsort(metrics)
    if minmax == 'min':
        best_index = sorted_indexes[0]
    elif minmax == 'max':
        best_index = sorted_indexes[-1]
    else:
        raise ValueError("Only min or max are allowed.")

    return data[best_index], metrics[best_index]


def quadratic_product(matrix: np.ndarray, center: np.ndarray, data: np.ndarray, mean: np.ndarray = None):
    # TODO: add checks
    if mean is not None:
        partial = np.matmul(matrix, mean - center).reshape(-1, 1)  # (dim, 1)
    else:
        partial = np.matmul(matrix, (data - center).T)  # (dim, npoints)
    return np.sum((data - center) * partial.T, axis=1)


if __name__ == '__main__':
    pass
