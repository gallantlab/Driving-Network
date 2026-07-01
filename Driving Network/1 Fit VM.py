# Fit voxelwise models for all 38 feature spaces to the data

from DrivingNetwork import IO, VM
from DrivingNetwork.DataInfo import Participants

# specify which participant to run on
participant = Participants.P1

# load data and features
train_data, test_data = IO.load_brain_data(participant)
train_features, test_features = IO.load_features(participant)

# make kernels for banded ridge regression
train_kernels, test_kernels = VM.make_kernels(train_features, test_features)

# fit model
performance, split_r2, dual_weights, kernel_weightings = VM.fit_model(train_kernels, train_data, test_kernels,
                                                                      test_data)

# compute primal weights from the dual weights
primal_weights = VM.compute_primal_weights(dual_weights, kernel_weightings, train_features, split_r2)

# run permutation tests to compute prediction performance significance
significance = VM.compute_prediction_significance(dual_weights, test_kernels, kernel_weightings, test_data)

# save these results
results = VM.Results(participant, dual_weights, kernel_weightings, performance, split_r2, primal_weights, significance)
IO.save_VM_results(results)