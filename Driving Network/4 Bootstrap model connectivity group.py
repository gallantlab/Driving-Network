# Bootstrap the model connectivity analysis for the whole group of subjects

import numpy
from DrivingNetwork import Clustering, IO, utils
from DrivingNetwork.DataInfo import Participants, FeatureSpaces
from DrivingNetwork.Config import BOOTSTRAP_CLUSTERING_PATH_TEMPLATE, BOOTSTRAP_CLUSTERING_INFO_TEMPLATE, NUM_CLUSTERING_BOOTSTRAPS
from DrivingNetwork.modelconn import linkage, fsaverage

fsaverage6 = fsaverage.fsaverageResolution.fsaverage6
fsaverage_converters = IO.load_individual_to_fsaverage_projections()

num_participants = len(Participants)

# load VM results
results = {}
for individual_weights_in_fsaverage in Participants:
    results[individual_weights_in_fsaverage] = IO.load_VM_results(individual_weights_in_fsaverage)

# Compute group PCs for weights from each feature space
group_weights = []
for i in range(len(FeatureSpaces)):
    group_weights.append(utils.to_group_average({participant: results[participant].primal_weights[i] for participant in Participants}))
group_weight_PCAs = Clustering.make_weight_PCA_objects(group_weights)

# project individual subjects weights to group space
# and keep them separate so the participants can be bootstrapped
individual_PCA_weights = {}
individual_weights_on_fsaverage = {}
for individual_weights_in_fsaverage in Participants:
    weight_PCs = []
    for i in range(len(FeatureSpaces)):
        if group_weight_PCAs[i] is None:
            weight_PCs.append(results[individual_weights_in_fsaverage].primal_weights[i])
        else:
            weight_PCs.append(group_weight_PCAs[i].transform(results[individual_weights_in_fsaverage].primal_weights[i].transpose()))
    individual_PCA_weights[individual_weights_in_fsaverage] = \
        fsaverage_converters[individual_weights_in_fsaverage].to_fsaverage(numpy.array(weight_PCs))
    individual_weights_on_fsaverage[individual_weights_in_fsaverage] = \
        fsaverage_converters[individual_weights_in_fsaverage].to_fsaverage(numpy.array(results[individual_weights_in_fsaverage].primal_weights))

allWeights = {'full': individual_weights_on_fsaverage,
              'PCA':  individual_PCA_weights}

# project each participant's significance mask to fsaverage6, kept separate so the participants can be bootstrapped
significances = {}
for participant in Participants:
    significances[participant] = fsaverage_converters[participant].to_fsaverage(
        results[participant].significance.astype(float), fsaverage6)

for i in range(NUM_CLUSTERING_BOOTSTRAPS):
    linkage_save_path = BOOTSTRAP_CLUSTERING_PATH_TEMPLATE.format('group', i)
    linkage_info_save_path = BOOTSTRAP_CLUSTERING_INFO_TEMPLATE.format('group', i)

    # pick random participants with replacement
    # choose weight type, and metric
    weight_type = ['full', 'PCA'][numpy.random.randint(2)]
    metric = [linkage.Metrics.Correlation, linkage.Metrics.Euclidean][numpy.random.randint(2)]
    participant_picks = [list(Participants)[numpy.random.choice(num_participants)] for i in range(num_participants)]

    # calculate group weights for these bootstrapped participants
    significance = numpy.zeros(int(fsaverage6) * 2)
    these_group_weights = numpy.zeros_like(allWeights[weight_type][participant_picks[0]])
    for subject in participant_picks:
        these_group_weights += allWeights[weight_type][subject]
        significance += significances[subject]
    these_group_weights /= 6
    these_group_weights = these_group_weights.transpose() # move vertex to 0th index

    # choose features with replacement
    num_features = these_group_weights.shape[1]
    feature_bootstrap_selection_indices = numpy.random.choice(num_features, num_features)
    these_group_weights = these_group_weights[:, feature_bootstrap_selection_indices]
    significance = significance > 2.5

    info = {'weight_type': 	weight_type,
            'metric': 		metric,
            'subjects': 	participant_picks,
            'feature_bootstrap_selection_indices': feature_bootstrap_selection_indices}

    # fit and save model connectivity results
    connectivity = linkage.Linkage('fsaverage', brainspace = fsaverage6)
    connectivity.fit(these_group_weights, metric, significance)
    connectivity.save(linkage_save_path, params = info)
    numpy.savez(linkage_info_save_path, **info)
