from typing import List, Iterator
import numpy
import joblib

from himalaya.kernel_ridge._random_search import solve_multiple_kernel_ridge_random_search
from himalaya.kernel_ridge._predictions import predict_and_score_weighted_kernel_ridge, predict_weighted_kernel_ridge
from himalaya.kernel_ridge._predictions import primal_weights_weighted_kernel_ridge
from himalaya.scoring import r2_score as R2
from himalaya.scoring import r2_score_split as SplitR2
from himalaya.backend import set_backend as SetBackend

from .DataInfo import Participants

Backend = SetBackend('torch')

class Results:
    """
    Basic struct to hold results from VM
    """
    def __init__(self, participant: Participants,
                 dualWeights: numpy.ndarray = None, deltas: numpy.ndarray = None,
                 performance: numpy.ndarray = None, splitR2: numpy.ndarray = None,
                 primalWeights: List[numpy.ndarray] = None, significance: numpy.ndarray = None) -> None:
        """
        Constructor
        :param participant:     participant this was fit on
        :param dualWeights:     weights in dual space from himalaya
        :param deltas:          kernel weightings from the himalaya fit
        :param performance:     overall R2 in predicted brain activity in each voxel
        :param splitR2:         split R2 for each feature space in each voxel
        :param primalWeights:   weights in primal space
        :param significance:    significance of the overall R2 in each voxel
        """
        self.participant = participant
        self.dual_weights = dualWeights
        self.deltas = deltas
        self.performance = performance
        self.split_R2 = splitR2
        self.primal_weights = primalWeights
        self.significance = significance

    def save(self, path: str) -> None:
        """
        Save this object to disk
        :param path:
        :return:
        """
        joblib.dump(self, path)


    @staticmethod
    def load(path:str) -> 'Results':
        """
        Loads a result object from disk
        :param path:
        :return:
        """
        return joblib.load(path)


def make_kernels(train_features: List[numpy.ndarray], test_features: List[numpy.ndarray]) -> tuple:
    """
    Makes kernels for the feature matrices for fitting in dual space

    :param train_features:  train feature matrices
    :param test_features:   test feature matrices
    :return: train, test feature kernels
    """
    train_kernels = numpy.array([numpy.dot(features, features.transpose()) for features in train_features])
    test_kernels = numpy.array([numpy.dot(test, train.transpose()) for test, train in zip(test_features, train_features)])
    traces = numpy.array([numpy.diag(kernel).sum() / len(kernel) for kernel in train_kernels])
    train_kernels = train_kernels / traces[:, numpy.newaxis, numpy.newaxis]
    test_kernels = test_kernels / traces[:, numpy.newaxis, numpy.newaxis]
    return train_kernels, test_kernels


def fit_model(train_kernels: numpy.ndarray, train_data: numpy.ndarray,
			  test_kernels: numpy.ndarray, test_data: numpy.ndarray) -> tuple:
    """
    Use himalaya to fit a VM model to the data. Hyperparameters are chosen by cross-validation on the
    train data, and model prediction performance is calculated on the test data.

    :param train_kernels:       train feature kernels
    :param train_data:          train data
    :param test_kernels:        test feature kernels
    :param test_data:           test data
    :return:
    """

    train_kernels = Backend.asarray(train_kernels)
    test_kernels = Backend.asarray(test_kernels)

    train_data = Backend.asarray(train_data)
    test_data = Backend.asarray(test_data)

    kernel_weightings, weights, cross_validation_performances = \
		solve_multiple_kernel_ridge_random_search(train_kernels, train_data, 2000,
												  [0.01, 0.1, 0.2, 0.5, 1.0, 2.0],
												  numpy.logspace(-3, 20, 21),
												  20, Ks_in_cpu = True,
												  n_targets_batch_refit = 10000,
												  return_weights = 'dual')
    performance = Backend.to_numpy(predict_and_score_weighted_kernel_ridge(test_kernels, weights, kernel_weightings,
                                                                           test_data, R2, False))
    split_r2 = Backend.to_numpy(predict_and_score_weighted_kernel_ridge(test_kernels, weights, kernel_weightings,
                                                                        test_data, SplitR2, True))

    return performance, split_r2, weights, kernel_weightings


def collapsed_delayed_feature_weights(weights: numpy.ndarray) -> numpy.ndarray:
    """
    The features are delayed across 5 TRs; this collapses the weights across TRs for each feature, so
    there's a single feature-to-voxel weight.

    :param weights:     primal weights with delays
    :return: average primal weight across the delays
    """
    nFeatures = int(weights.shape[0] / 5)

    out = numpy.zeros([nFeatures, weights.shape[1]])

    for i in range(5):
        out += weights[(i * nFeatures):((i + 1) * nFeatures), :]
    out /= 5

    return out


def compute_primal_weights(dual_weights: numpy.ndarray, kernel_weightings: numpy.ndarray,
						   train_features: List[numpy.ndarray], split_R2: numpy.ndarray,
						   epsilon: float = 1e-6) -> List[numpy.ndarray]:
    """
    Compute primal weight from dual weights. These primal weights are then rescaled by the R2 score.

    :param dual_weights:        fitted dual weights
    :param kernel_weightings:   fitted kernel weightings for the dual weights
    :param train_features:      train features
    :param split_R2:            split R2 scores
    :param epsilon:             R2 values below which to zero weights out
    :return:
    """
    primal = primal_weights_weighted_kernel_ridge(dual_weights, kernel_weightings, train_features)
    collapsed = [collapsed_delayed_feature_weights(weights) for weights in primal]

    for i, weights in enumerate(collapsed):
        performance = split_R2[i, :]
        reNorm = numpy.clip(performance / numpy.percentile(performance, 99), 0, 1)
        for voxel in range(weights.shape[1]):
            norm = numpy.linalg.norm(weights[:, voxel])
            if (norm > epsilon) and (performance[voxel] > epsilon):
                weights[:, voxel] /= norm
                weights[:, voxel] *= reNorm[voxel]
            else:
                weights[:, voxel] = 0
    return collapsed


def compute_prediction_significance(dual_weights: numpy.ndarray, test_kernels: numpy.ndarray,
									kernel_weightings: numpy.ndarray, test_data: numpy.ndarray,
									num_iterations: int = 1000, block_size: int = 20,
									alpha: float = 0.05) -> numpy.ndarray:
    """
    Computes significance for prediction performance using permutation test, and with Benjamini-Hochberg correction
    :param dual_weights:        dual weights from fit model
    :param test_kernels:        test feature kernels
    :param kernel_weightings:   kernel weights from fit model
    :param test_data:           test data against which to compare
    :param num_iterations:      number of permutation test iterations
    :param block_size:          block size for permutations, approximate if it doesn't divide evenly into the number of TRs
    :param alpha:               significance threshold
    :return:    binary mask of which voxels are significant
    """
    predictions = predict_weighted_kernel_ridge(test_kernels, dual_weights, kernel_weightings)
    score = R2(test_data, predictions)
    permuted_scores = []

    chunks = numpy.array_split(predictions, int(predictions.shape[0] / block_size), axis = 0)
    for i in range(num_iterations):
        permuted = numpy.vstack(numpy.random.permutation(chunks))
        permuted_scores.append(R2(test_data, permuted))
    permuted_scores = numpy.array(permuted_scores)

    num_voxels = score.shape[0]
    counts = numpy.zeros(num_voxels)
    for v in range(num_voxels):
        counts[v] = numpy.sum(score[v] < permuted_scores[:, v])
    counts /= num_iterations

    sorted = counts.copy()
    sorted.sort()
    thresholds = (numpy.arange(num_voxels) + 1) / num_voxels * alpha

    corrected_alpha = sorted[numpy.max(numpy.where(sorted < thresholds))]

    return counts < corrected_alpha