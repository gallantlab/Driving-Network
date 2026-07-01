# We use model connectivity to identify the network of cortical regions
# that support active navigation
# Because clustering solutions are sensitive to noise and parameter choices, we bootstrap the clustering
# This bootstraps the model connectivity analysis in an individual subject
import numpy

from DrivingNetwork import Clustering, IO
from DrivingNetwork.DataInfo import Participants, FeatureSpaces
from DrivingNetwork.Config import BOOTSTRAP_CLUSTERING_PATH_TEMPLATE, BOOTSTRAP_CLUSTERING_INFO_TEMPLATE, \
                                  NUM_CLUSTERING_BOOTSTRAPS
from DrivingNetwork.modelconn import linkage

participant = Participants.P1

results = IO.load_VM_results(participant)

# load weights
full_weights = results.primal_weights                   # weights for all features in all voxels
weight_PCs = Clustering.get_weights_PCs(full_weights)   # reduced weights - any feature space larger than 10 features is truncated to 10 PCs
significance_mask = results.significance

# mask weights down to significant voxels
weights = {'full': numpy.vstack(full_weights).transpose()[significance_mask, :],
           'PCA':  numpy.vstack(weight_PCs).transpose()[significance_mask, :]}

# bootstrap the clustering used in MC
# on each bootstrap, we randomly pick between using the full weights, or the top 10 PCs of each feature spaces
# in the selected weights, we bootstrap the weight dimensions that are used
# we also randomly pick between correlation and euclidean distance
for i in range(NUM_CLUSTERING_BOOTSTRAPS):
    # save location for this bootstrap
    linkage_save_path = BOOTSTRAP_CLUSTERING_PATH_TEMPLATE.format(participant.name, i)
    linkage_info_save_path = BOOTSTRAP_CLUSTERING_INFO_TEMPLATE.format(participant.name, i)

    # pick random weight type and metric
    weight_type = ['full', 'PCA'][numpy.random.randint(2)]
    metric = [linkage.Metrics.Correlation, linkage.Metrics.Euclidean][numpy.random.randint(2)]

    # choose feature dimensions with replacement
    original_weights = weights[weight_type]
    num_features = original_weights.shape[1]
    feature_bootstrap_selection_indices = numpy.random.choice(num_features, num_features)
    bootstrapped_weights = original_weights[:, feature_bootstrap_selection_indices]

    # record what was chosen on this bootstrap
    info = {'weight_type': 	weight_type,
            'metric': 		metric,
            'feature_bootstrap_selection_indices': feature_bootstrap_selection_indices}

    # fit model connectivity to this bootstrap and save out results
    connectivity = linkage.Linkage(participant.name)
    connectivity.fit(bootstrapped_weights, metric, significance_mask)
    connectivity.save(linkage_save_path)            # save actual linkage
    numpy.savez(linkage_info_save_path, **info)     # save parameters on this bootstrap iteration