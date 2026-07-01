# We use the bootstrapped  model connectivity solutions to calculate a navigation
# preference index in a single subject. The NPI reflects how often, across bootstrapped
# model connectivity solutions, each voxel is sorted into the network in which the
# navigation models explain the most variance. This makes the MC process more robust
# to noise and clustering parameter choices.
from DrivingNetwork import Clustering, IO
from DrivingNetwork.DataInfo import Participants
from DrivingNetwork.Config import BOOTSTRAP_CLUSTERING_PATH_TEMPLATE, NUM_CLUSTERING_BOOTSTRAPS
from DrivingNetwork.modelconn.linkage import Linkage

import numpy

participant = Participants.P1
results = IO.load_VM_results(participant)

# load the boostrapped model connectivity solutions
bootstraps = []
for i in range(NUM_CLUSTERING_BOOTSTRAPS):
    bootstraps.append(Linkage.load(BOOTSTRAP_CLUSTERING_PATH_TEMPLATE.format(participant.name, i)))

navigation_scores = []
# use 1000 samplings of the bootstraps to compute a NPI
for i in range(1000):
    # pick a random clustering bootstrap
    solution = bootstraps[numpy.random.randint(NUM_CLUSTERING_BOOTSTRAPS)]
    # pick a random number of clusters
    num_clusters = int(numpy.exp2(numpy.random.rand() * 12 + 2)) # range of 4, 16384 clusters with a logarithmic bias
    # get the clustering solution for this clustering
    clustering = solution.get_clustering(num_clusters, unmask = True)
    # record which voxels are best-predicted
    navigation_scores.append(Clustering.get_navigation_score_for_one_clustering(clustering, results.split_R2))

# the NPI is the frequency each voxel is sorted into the best-explained cluster by the navigation models
NPI = numpy.sum(numpy.array(navigation_scores), axis = 0) / 1000