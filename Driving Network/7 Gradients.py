# Visualize the functional organization of the cortex with UMAP, and quantify the
# posterior-anterior concreteness gradient.

import umap
import numpy

from DrivingNetwork import IO, utils, Clustering, Plotting
from DrivingNetwork.DataInfo import Participants, FeatureSpaces

# load the fit models
converters = IO.load_individual_to_fsaverage_projections()
results = IO.load_all_results()
group_split_R2 = IO.load_group_split_R2(results, converters)
group_performance = utils.to_group_average(
	{participant: results[participant].performance for participant in Participants}, converters)
significance_count = IO.load_group_significance_count(results, converters)

# prep weights for UMAP
group_weights = [utils.to_group_average(
	{participant: results[participant].primal_weights[i] for participant in Participants}, converters)
	for i in range(len(FeatureSpaces))]
weight_PCAs = Clustering.make_weight_PCA_objects(group_weights)
reduced_weights = []
for i in range(len(FeatureSpaces)):
	if weight_PCAs[i] is None:
		reduced_weights.append(group_weights[i])
	else:
		reduced_weights.append(weight_PCAs[i].transform(group_weights[i].transpose()).transpose())
PCA_weights = numpy.vstack(reduced_weights).transpose()

navigation_masks, comparison_masks, navigation_union = utils.load_roi_masks()

# Create a UMAP projection of all the vertices on the brain from their weights.
projector = umap.UMAP(n_neighbors = 256, min_dist = 0.5, n_components = 2)
projector.fit(PCA_weights[significance_count > 2, :])
reduced = projector.transform(PCA_weights)

# plot UMAP functional distribution of all vertices
umap_figure = Plotting.make_umap_roi_figure(reduced, navigation_masks, comparison_masks)

# Map UMAP functional distribution for cortical surface
upscaler = IO.load_fsaverage_upscaler()
flatmap = Plotting.make_umap_flatmap(reduced, group_performance, significance_count, upscaler)

# plot concreteness gradients
concreteness = utils.compute_concreteness_index(group_split_R2)
posterior_anterior_scale = utils.compute_posterior_anterior_scale()

gradient_slopes = utils.compute_roi_concreteness_slopes(navigation_masks, posterior_anterior_scale,
														concreteness, significance_count)

individual_roi_concreteness = Plotting.make_concreteness_figure_individual_rois(gradient_slopes)
all_roi_concreteness = Plotting.make_concreteness_figure_all_rois(posterior_anterior_scale, concreteness,
																  navigation_union, significance_count)
