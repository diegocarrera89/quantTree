import numpy as np
from quanttree import utils as ccm
from abc import ABC, abstractmethod
from sklearn.decomposition import PCA


# Function to give data the proper shape
def reshape_data(data) -> np.ndarray:
    data = np.array(data)
    if len(data.shape) == 1:
        data = data[:, np.newaxis]
    return data


# Total Variation statistic
def tv_statistic_(_pi_exp, _pi_hats, _nu):
    # _pi_exp.shape = (K,) or (nbatches, K)
    # _pi_hats.shape == (nbatches, K)
    return 0.5 * np.sum(_nu * np.abs(_pi_exp - _pi_hats), axis=1)


# Pearson statistic
def pearson_statistic_(_pi_exp, _pi_hats, _nu, tol=10 ** (-12)):
    # _pi_exp.shape = (K,) or (nbatches, K)
    # _pi_hats.shape == (nbatches, K)
    return np.sum(_nu * ((_pi_exp - _pi_hats) ** 2) / _pi_exp, axis=1)


# --- ABSTRACT CLASSES ---

# Abstract class representing a transformation of the data.
# The subclasses has to implement two methods:
# - estimate_transformation: estimates the parameter of the transformation if necessary, e.g. in case of PCARotation.
#       It takes as input the training data
# - transform_data: it actually transforms the input data and returns the transformed data
# It has a concrete factory method for each implemented subclasses, i.e.:
# - Identity: it does not perform any transformation of data
# - RandomRotation: it rotates the data w.r.t. a random rotation matrix generated at initialization time
# - PCARotation: based on the PCA estimated from training data
class DataTransformation(ABC):

    @abstractmethod
    def estimate_transformation(self, data: np.ndarray):
        pass

    @abstractmethod
    def transform_data(self, data: np.ndarray):
        pass

    @classmethod
    def get_data_transformation(cls, transformation_type: str = 'none'):
        if str.lower(transformation_type) == 'none':
            return Identity()
        if str.lower(transformation_type) == 'pca':
            return PCARotation()
        if str.lower(transformation_type) == 'random':
            return RandomRotation()


# Abstract class representing a partitioning of data space. Each subclass has to implement two (protected) methods:
# - _build_partitioning: given a training set, it builds the partitioning og the data space
# - _find_bin: given a batch of data it returns the indices of the bin for each data
# The constructor of Partitioning class takes as input a DataTransformation object. The Partitioning class transforms
# training and test data according to this transformation (the methods _find_bin and _build_partitioning DO NOT have to
# implement the tranformation of the data)
class Partitioning(ABC):

    def __init__(self, nbin: int = 2, transformation_type: str = 'none'):
        self.nbin: int = nbin
        self.transformation: DataTransformation = DataTransformation.get_data_transformation(transformation_type)

    @abstractmethod
    def _build_partitioning(self, data):
        pass

    @abstractmethod
    def _find_bin(self, data):
        pass

    @abstractmethod
    def _get_bin_counts(self, data):
        bins = self._find_bin(data)
        return np.bincount(bins, minlength=self.nbin)

    def build_partitioning(self, data):
        data = reshape_data(data)
        self.transformation.estimate_transformation(data)
        data = self.transform_data(data)
        self._build_partitioning(data)

    def find_bin(self, data):
        data = self.transform_data(data)
        return self._find_bin(data)

    def get_bin_counts(self, data):
        data = self.transform_data(data)
        return self._get_bin_counts(data)

    def transform_data(self, data):
        return self.transformation.transform_data(data)


# Abstract class representing a model for data. It has two methods: one for training (train_model) and one to assess the
# goodness of fit of test data to the learned model
class DataModel(ABC):

    @abstractmethod
    def assess_goodness_of_fit(self, data: np.ndarray):
        pass

    @abstractmethod
    def train_model(self, data: np.ndarray):
        pass


class ThresholdStrategy(ABC):

    def __init__(self):
        self.threshold_dict = dict()

    @abstractmethod
    def get_threshold(self, alpha):
        pass

    @abstractmethod
    def add_threshold(self):
        pass

    # non necessario da implementare, ma solo se serve configurarla su un modello
    def configure_strategy(self, model: DataModel, data_training: np.ndarray):
        pass


# --- Independent classes ---

class Statistic:

    def __init__(self, statistic_name: str = 'pearson', pi_values=None):
        if statistic_name == 'pearson':
            self.statistic = pearson_statistic_
        elif statistic_name == 'tv':
            self.statistic = tv_statistic_
        else:
            raise Exception(f"Invalid statistic name: {statistic_name}")

        # Reference distribution probability
        self.pi_values = pi_values

    def compute_statistic(self, pi_hats, nu):
        return self.statistic(_pi_exp=self.pi_values, _pi_hats=pi_hats, _nu=nu)


# --- Concrete classes ---

# Class representing a histogram as data model. The histogram is defined through a Partitioning object, while the
# goodness of fit is assess through the Pearson statistic or total variation distance (the choice among the two is made
# at contruction time.
class Histogram(DataModel):

    def __init__(self, partitioning: Partitioning, statistic_name: str = 'pearson'):
        # Underlying space partitioning
        self.partitioning: Partitioning = partitioning

        # Statistic name and instance
        self.statistic_name: str = statistic_name
        self.statistic: Statistic = Statistic()

        # Uniformly initialized pi_values
        self.pi_values: np.ndarray = np.ones(self.partitioning.nbin) * 1 / self.partitioning.nbin

    def __hash__(self):
        tmp = (self.statistic_name.__hash__(), self.partitioning.__hash__(), tuple(self.pi_values).__hash__())
        return tmp.__hash__()

    def __eq__(self, other):
        if not isinstance(other, Histogram):
            return False
        else:
            return self.statistic_name == other.statistic_name and self.partitioning == other.partitioning and tuple(
                self.pi_values) == tuple(other.pi_values)

    def assess_goodness_of_fit(self, data):
        data = reshape_data(data)
        if len(data.shape) == 2:
            data = data.reshape((1,) + data.shape)
        _, nu, _ = data.shape
        pi_hats = self.partitioning.get_bin_counts(data) / nu
        if len(pi_hats.shape) == 1:
            pi_hats = pi_hats.reshape((1,) + pi_hats.shape)
        return self.statistic.compute_statistic(pi_hats=pi_hats, nu=nu)

    def train_model(self, data):
        self.partitioning.build_partitioning(data)
        self.estimate_probabilities(data)
        self.set_statistic()

    def estimate_probabilities(self, data: np.ndarray):
        data = reshape_data(data)
        self.pi_values = self.get_bin_counts(data) / data.shape[0]

    def get_bin_counts(self, data: np.ndarray):
        data = reshape_data(data)
        return self.partitioning.get_bin_counts(data)

    def set_statistic(self):
        self.statistic = Statistic(statistic_name=self.statistic_name, pi_values=self.pi_values)


# DataTransformation
class RandomRotation(DataTransformation):

    def __init__(self):
        super().__init__()
        self.rotation = []

    def estimate_transformation(self, data: np.ndarray):
        data = reshape_data(data)
        dim = data.shape[1]
        self.rotation = ccm.generate_random_rotation(dim)

    def transform_data(self, data):
        data = reshape_data(data)
        data_new = np.dot(data, self.rotation)
        return data_new


class PCARotation(DataTransformation):

    def __init__(self):
        super().__init__()
        self.pc = self.pc = PCA(whiten=False)

    def estimate_transformation(self, data: np.ndarray):
        data = reshape_data(data)

        if len(data) == 3:
            data = data.reshape((-1, data.shape[2]))

        self.pc.fit(data)

    def transform_data(self, data):
        data = reshape_data(data)

        if len(data.shape) == 2:
            return self.pc.transform(data)
        elif len(data.shape) == 3:
            d0, d1, d2 = data.shape
            data = data.reshape((-1, d2))
            return self.pc.transform(data).reshape((d0, d1, d2))


class Identity(DataTransformation):

    def __init__(self):
        super().__init__()

    def estimate_transformation(self, data: np.ndarray):
        pass

    def transform_data(self, data):
        return data
