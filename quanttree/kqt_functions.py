import numpy as np


def linear_kernel(points, node):
    return np.dot(points, node)


def polynomial_kernel(points, node, residual, degree):
    return (np.dot(points, node) + residual) ** degree


def gaussian_kernel(points, node, sigma):
    return np.exp(- np.linalg.norm(points - node, axis=-1) ** 2 / (2 * sigma ** 2))


def laplacian_kernel(points, node, alpha):
    return np.exp(- alpha * np.linalg.norm(points - node, axis=-1))


def get_kernel(kernel_type):
    if kernel_type == "linear":
        return linear_kernel
    elif kernel_type == "polynomial":
        return polynomial_kernel
    elif kernel_type == "gaussian":
        return gaussian_kernel
    elif kernel_type == "laplacian":
        return laplacian_kernel
    else:
        error_string = [f"Invalid kernel type: {kernel_type}",
                        f"Valid kernel types: linear, polynomial, gaussian, laplacian."]
        raise TypeError('\n'.join(error_string))


class KernelFunction:
    def __init__(self, kernel_type, **kwargs):
        self.kernel_type = kernel_type
        self.kernel_parameters = kwargs
        self.kernel = get_kernel(kernel_type=kernel_type)

    def __str__(self):
        return f"KernelFunction({self.kernel_type}, {self.kernel_parameters})"

    def __call__(self, points, node=None):
        if node is None:
            node = np.zeros(points.shape[-1])
        return self.compute(points, node)

    def compute(self, points, node):
        if len(self.kernel_parameters) == 0:
            return self.kernel(points=points, node=node)
        else:
            return self.kernel(points=points, node=node, **self.kernel_parameters)
