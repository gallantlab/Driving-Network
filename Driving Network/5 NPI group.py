# calculate the navigation preference index at the group level
# using the bootstrapped model connectivity solutions

import numpy
from DrivingNetwork import Clustering, IO, utils
from DrivingNetwork.DataInfo import Participants
from DrivingNetwork.Config import BOOTSTRAP_CLUSTERING_PATH_TEMPLATE, NUM_CLUSTERING_BOOTSTRAPS
from DrivingNetwork.modelconn.linkage import Linkage

# load VM results
results = {}
for participant in Participants:
	results[participant] = IO.load_VM_results(participant)
group_split_R2 = utils.to_group_average({participant: results[participant].split_R2 for participant in Participants})

# load bootstrapped model connectivity solutions
bootstraps = []
for i in range(NUM_CLUSTERING_BOOTSTRAPS):
	bootstraps.append(Linkage.load(BOOTSTRAP_CLUSTERING_PATH_TEMPLATE.format('group', i)))

# use 1000 clusterings to calculate a navigation preference index
navigation_scores = []
for i in range(1000):
	solution = bootstraps[numpy.random.randint(NUM_CLUSTERING_BOOTSTRAPS)]
	num_clusters = int(numpy.exp2(numpy.random.rand() * 12 + 2)) # range of 4, 16384 clusters with a logarithmic bias
	clustering = solution.get_clustering(num_clusters, unmask = True)
	navigation_scores.append(Clustering.get_navigation_score_for_one_clustering(clustering, group_split_R2))

NPI = numpy.sum(numpy.array(navigation_scores), axis = 0) / 1000