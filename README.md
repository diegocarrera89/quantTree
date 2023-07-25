# Introduction
This repository contains the Python implementation of QuantTree \[[Boracchi et al. 2018](#boracchi-et-al-2018)\] and its extensions QT-EWMA \[[Frittoli et al. 2021](#frittoli-et-al-2021)\]\[[Frittoli et al. 2022](#frittoli-et-al-2022)\], CDM \[[Stucchi et al. 2022](#stucchi-et-al-2022)\] and Kernel QuantTree \[[Stucchi et. al. 2023](#stucchi-et-al-2023)\].

# Dependencies
Python 3 with the packages in [requirements.txt](requirements.txt)

# Install
1. Clone the repository
2. Run `pip install -e path/to/quanttree`

# Brief description
In this section, we illustrate the methods that are implemented in this library. For in-depth explanation, we recommend checking out the papers associated with each algorithm.

## QuantTree
QuantTree monitors batches of data to detect any distribution change $\phi_0 \to \phi_1$ affecting the batches. During training, a histogram is constructed over a training set. Each bin of the histogram is defined by _i)_ projecting training data on a random dimension and _ii)_ computing a quantile of the projected samples to isolate the pre-defined percentage of points that fall in the bin. The detection threshold is computed via Monte Carlo simulations, as described in \[[Boracchi et al. 2018](#boracchi-et-al-2018)\] and \[[Frittoli et al. 2022](#frittoli-et-al-2022)\]. During testing, a batch is mapped to a bin-probability vector by counting the number of batch samples falling in each bin. Then, the test statistic associated with the batch only depends on its bin-probability vector. A change is detected when the test statistic exceeds the threshold.

The main parameters of QuantTree are:

- the number of bins $K$;
- the desired percentage of points per bin $\{\pi_1, ..., \pi_K\}$;
- the target False Positive Rate $\alpha$;
- the test statistic to be employed.

The QuantTree is implemented in `quanttree/quanttree_src.py` in a class called `QuantTree`. We refer to the inline documentation of `QuantTree` for a detailed explanation of the `__init__` arguments. In `demos/demo_quanttree.py`, you can see QuantTree in action (with comments!).

## QT-EWMA
QuantTree Exponentially Weighted Moving Average (QT-EWMA) monitors datastreams by means of an online statistical test that monitors the bin frequencies a QuantTree histogram by EWMA statistics. During training, a QuantTree histogram is constructed over the training set drawn from $\phi_0$. During testing, each incoming sample is associated with the bin it falls into, and the EWMA statistics are updated accordingly. Then, the QT-EWMA test statistic combines the differences between the EWMA statistics and their expected values under $\phi_0$. Detection thresholds are computed by Monte Carlo simulations (as detailed in \[[Frittoli et al. 2021](#frittoli-et-al-2021)\]\[[Frittoli et al. 2022](#frittoli-et-al-2022)\]) to guarantee the desired Average Running Length (ARL0), namely, the expected time before a false alarm. 

The main parameters of QT-EWMA are:

- the number of bins $K$;
- the target ARL0;
- the EWMA forgetting factor $\lambda$;

QT-EWMA is implemented in `quanttree/qtewma_src.py` in a class called `QT_EWMA`. We refer to the inline documentation of `QT_EWMA` for a detailed explanation of the `__init__` arguments. In `demos/demo_qtewma.py`, QT-EWMA is used in a small experiment over synthetic data.

## QT-EWMA-update
QT-EWMA-update is a change-detection algorithm based on QT-EWMA that enables online monitoring even when the training set is extremely small. In QT-EWMA-update, new samples are used to update the estimated bin probabilities of the initial QuantTree histogram (namely, the estimated expected values of the EWMA statistics), as long as no change is detected. This update improves the model, thus increasing the detection power. The updating procedure is compatible with the computational requirements of online monitoring schemes, and the distribution of the QT-EWMA-update statistic is also independent of the stationary distribution, enabling the computation of thresholds controlling the ARL0 through the same procedure as in QT-EWMA.

To set up QT-EWMA-update, the following parameters are required:

- the number of bins $K$;
- the target ARL0;
- the EWMA forgetting factor $\lambda$;
- the weight of the latest sample during the update $\beta$;
- the number of samples after which the update stops $S$ (optional);

QT-EWMA-update can be used by setting the correct parameters in the initialization of a `QT_EWMA` instance.

## CDM
Class Distribution Monitoring (CDM) employs separate instances of QT-EWMA to monitor the class-conditional distributions. We report a concept drift after detecting a change in the class-conditional distribution of at least one class. The main advantages of CDM are:

i) it can detect any relevant drift, including virtual ones that have little impact on the classification error and are by design ignored by methods that monitor the error rate;
ii) it can detect concept drifts affecting only a subset of classes more promptly than methods that monitor the overall data distribution, since the other class-conditional distributions do not change;
iii) it provides insights on which classes have been affected by concept drift, which might be crucial for diagnostics and adaptation;
iv) it effectively controls false alarms by maintaining a target ARL0, set before monitoring.

The setup of CDM simply consists of setting the parameters for the underlying QT-EWMA (see previous sections).

CDM is implemented in `quanttree/cdm_src.py` in a class called `CDM_QT_EWMA`. We refer to the inline documentation of `CDM_QT_EWMA` for a detailed explanation of the `__init__` arguments. In `demos/demo_cdm.py`, CDM is compared against QT-EWMA in an experiment over a synthetic datastream comprising two classes.

We remark that the control of the ARL0 holds for any CDM defined by any online change-detection algorithm that can be configured to yield the desired ARL0 by setting a constant false alarm probability over time (see Proposition 1 in \[[Stucchi et al. 2022](#stucchi-et-al-2022)\]). This means that, in principle, we can define CDM using other change-detection tests. However, to the best of our knowledge, QT-EWMA is the only nonparametric and online change-detection test for multivariate datastreams where the ARL0 is controlled by setting a constant false alarm probability.

## Kernel QuantTree
Kernel QuantTree monitors batches of data to detect any distribution change $\phi_0 \to \phi_1$ affecting the batches. As in QuantTree, during training, a histogram is constructed over a training set. Each bin of the histogram is defined by _i)_ mapping multivariate training data to the univariate space via measurable kernel functions and _ii)_ computing a quantile of the projected samples to isolate the pre-defined percentage of points that fall in the bin. In contrast with QuantTree, the bins of Kernel QuantTree are compact subsets of the input space with finite volume. The detection threshold is computed via Monte Carlo simulations, as described in \[[Boracchi et al. 2018](#boracchi-et-al-2018)\] and \[[Frittoli et al. 2022](#frittoli-et-al-2022)\]. During testing, a batch is mapped to a bin-probability vector by counting the number of batch samples falling in each bin. Then, the test statistic associated with the batch only depends on its bin-probability vector. A change is detected when the test statistic exceeds the threshold.

The main parameters of Kernel QuantTree are:

- the kernel functions $f_k$ to be employed;
- the number of bins $K$;
- the desired percentage of points per bin $\{\pi_1, ..., \pi_K\}$;
- the target False Positive Rate $\alpha$;
- the test statistic to be employed.

The Kernel QuantTree is implemented in the following files:

- `quanttree/kqt_eucliean.py` in a class called `EuclideanKernelQuantTree`;
- `quanttree/kqt_mahalanobis.py` in a class called `MahalanobisKernelQuantTree`;
- `quanttree/kqt_weighted_mahalanobis.py` in a class called `WeightedMahalanobisKernelQuantTree`;

We refer to the inline documentation of each class for a detailed explanation of the `__init__` arguments. In `demos/demo_kqt.py`, you can see the proposed Kernel QuantTrees in action (with comments!).

## MultiModal QuantTree
Coming soon.

# Thresholds Computation
The theoretical properties of QuantTree enable an efficient monitoring where detection thresholds are independent of the stationary distribution, and can be pre-computed via Monte Carlo simulations, as detailed in the papers reported in the [References](#references). Here, we provide pre-computed thresholds for the settings addressed in the experimental sections of the works involving QuantTree and its extensions. For any question about thresholds, see [Contacts](#contacts)

# Contacts
The main contributors to this repo are Diego Stucchi, Diego Carrera and Luca Frittoli. For any question or bug report, please contact <diego.stucchi@polimi.it>.

# License
This software is released under a NonCommercial-ShareAlike license issued by Politecnico di Milano. The adaptation of this software is allowed under specific conditions that are designed to enable most non-commercial uses.
See [LICENSE.pdf](LICENSE.pdf) for the complete terms and conditions.


# References

###### [Boracchi et al. 2018] 
_"QuantTree: Histograms for Change Detection in Multivariate Data Streams"_  
G. Boracchi, D. Carrera, C. Cervellera, D. Macciò. International Conference on Machine Learning (ICML) 2018.

###### [Frittoli et al. 2021]
_"Change Detection in Multivariate Datastreams Controlling False Alarms"_  
L. Frittoli, D. Carrera, G. Boracchi. Joint European Conference on Machine Learning and Knowledge Discovery in Databases 2021.

###### [Frittoli et al. 2022]
_"Nonparametric and Online Change Detection in Multivariate Datastreams using QuantTree"_  
L. Frittoli, D. Carrera, G. Boracchi. IEEE Transactions on Knowledge and Data Engineering 2022.

###### [Stucchi et al. 2022]
_"Class Distribution Monitoring for Concept Drift Detection"_  
D. Stucchi, L. Frittoli, G. Boracchi. IEEE-INNS International Joint Conference on Neural Networks (IJCNN) 2022.

###### [Stucchi et al. 2023]
_"Kernel QuantTree"_
D. Stucchi, P. Rizzo, N. Folloni, G. Boracchi. International Conference on Machine Learning (ICML) 2023.
