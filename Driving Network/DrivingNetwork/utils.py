from typing import Optional, Dict
from .modelconn.fsaverage import Converter, fsaverageResolution

from . import IO
import numpy
from scipy import stats
from sklearn.linear_model import LinearRegression
import cortex

from .DataInfo import Participants, ROIs
from .Config import NAVIGATION_ROI_OVERLAY_PATH

def get_index(item: str, enumerable: list) -> int:
	for i in range(len(enumerable)):
		if (item == enumerable[i]):
			return i
	raise ValueError('Item not in enumerable')


def to_group_average(data: Dict[Participants, numpy.ndarray], converters: Optional[Dict[Participants, Converter]] = None,
					 resolution: fsaverageResolution = fsaverageResolution.fsaverage6) -> numpy.ndarray:
	"""
	Group-averages data in fsaverage space

	@param data: 		data to be averaged
	@param converters: 	fsaverage converters
	@param resolution: 	resolution to work in
	@return:
	"""
	if converters is None:
		converters = IO.load_individual_to_fsaverage_projections()
	projected = [converters[subject].to_fsaverage(data[subject], resolution) for subject in Participants]
	average = numpy.zeros_like(projected[0])
	for s in range(len(Participants)):
		average += projected[s]
	average /= len(Participants)
	return average


def fs7_ROI_to_fs6(mask: numpy.ndarray) -> numpy.ndarray:
	"""
	Reduces fsaverage7 ROI mask to fsaverage6

	:param mask: 	ROI mask in fsaverage7
	:return: ROI mask in fsaverage6
	"""
	hemisphere7 = int(fsaverageResolution.fsaverage7)
	hemisphere6 = int(fsaverageResolution.fsaverage6)
	left = mask[:hemisphere7][:hemisphere6]
	right = mask[hemisphere7:][:hemisphere6]
	return numpy.hstack([left, right])


def load_roi_masks() -> tuple:
	"""
	Load ROI masks in fsaverage6

	:return: navigation ROIs, other reference RIs, and the boolean union of the navigation ROIs
	"""

	comparison_roi_names = ['V1', 'V2', 'V3', 'IPS', 'FEF', 'M1H', 'S1H', 'M1F', 'S1F', 'M1M', 'S1M']

	navigation_masks_fs7 = cortex.get_roi_verts('fsaverage', ROIs, mask = True,
												overlay_file = NAVIGATION_ROI_OVERLAY_PATH)
	nav_rois = {roi: fs7_ROI_to_fs6(navigation_masks_fs7[roi]) for roi in ROIs}

	other_rois_fs7 = cortex.get_roi_verts('fsaverage', comparison_roi_names, mask = True)
	other_rois = {roi: fs7_ROI_to_fs6(other_rois_fs7[roi]) for roi in comparison_roi_names}

	navigation_union = numpy.zeros(int(fsaverageResolution.fsaverage6) * 2, dtype = bool)
	for roi in nav_rois:
		navigation_union = numpy.logical_or(navigation_union, nav_rois[roi])

	return nav_rois, other_rois, navigation_union


def compute_concreteness_index(group_split_R2: numpy.ndarray) -> numpy.ndarray:
	"""
	Computes the concreteness index for each vertex

	:param group_split_R2: 	group-average split R2 at each vertex
	:return: the concreteness index for each vertex, on [-1, 1]
	"""
	concrete = numpy.concatenate([numpy.ones(14), numpy.zeros(24)])
	abstract = numpy.concatenate([numpy.zeros(14), numpy.ones(24)])
	concrete_R2 = numpy.dot(concrete, group_split_R2)
	abstract_R2 = numpy.dot(abstract, group_split_R2)
	return (concrete_R2 - abstract_R2) / (concrete_R2 + abstract_R2)


def compute_posterior_anterior_scale() -> numpy.ndarray:
	"""
	Computes the geodesic posterior-anterior position of each fsaverage6 vertex.
	The position is on a [-1, 1] range in which -1 is the most posterior and +1 is the most anterior

	:return: the posterior-anterior position for each vertex
	"""
	import cortex
	from cortex import freesurfer
	surfaces = []
	for hemisphere in ['lh', 'rh']:
		vertex_coordinates, faces, _ = freesurfer.get_surf('fsaverage6', hemisphere, 'inflated')
		surfaces.append(cortex.polyutils.Surface(vertex_coordinates, faces))

	posterior_vertices = [numpy.argmin(surface.pts[:, 1]) for surface in surfaces]
	anterior_vertices = [numpy.argmax(surface.pts[:, 1]) for surface in surfaces]
	posterior_distances = [surfaces[i].geodesic_distance(posterior_vertices[i]) for i in [0, 1]]
	anterior_distances = [surfaces[i].geodesic_distance(anterior_vertices[i]) for i in [0, 1]]

	scale = []
	for i in range(2):
		posterior_distance = posterior_distances[i]
		anterior_distance = anterior_distances[i]
		scale.append((posterior_distance - anterior_distance) / (posterior_distance + anterior_distance))
	return numpy.concatenate(scale)


def compute_slope(position: numpy.ndarray, values: numpy.ndarray, num_bootstraps: int = 5000) -> tuple:
	"""
	Fit OLS for values against position to find the slope and bootstrap
	the 99% confidence interval for the regression line

	:param position: 		X
	:param values: 			Y
	:param num_bootstraps: 	number of bootstrap resamples
	:return: sorted position, regression line, CI lower bound, CI upper bound, and the slope
	"""
	correlation = stats.pearsonr(position, values)
	sorted_position = numpy.sort(position)

	predictions = []
	for _ in range(num_bootstraps):
		indices = numpy.random.randint(0, position.shape[0], position.shape[0])
		regressor = LinearRegression()
		regressor.fit(position[indices][:, None], values[indices][:, None])
		predictions.append(regressor.predict(sorted_position[:, None]).squeeze())
	predictions = numpy.array(predictions)
	lower = numpy.percentile(predictions, 0.5, axis = 0)
	upper = numpy.percentile(predictions, 99.5, axis = 0)

	regressor = LinearRegression()
	regressor.fit(position[:, None], values[:, None])
	line = regressor.predict(sorted_position[:, None]).squeeze()
	return sorted_position, line, lower, upper, correlation


def compute_roi_concreteness_slopes(roi_masks: dict, position: numpy.ndarray, concreteness: numpy.ndarray,
									significance_count: numpy.ndarray, num_bootstraps: int = 5000) -> dict:
	"""
	Computes the concreteness-against posterior/anterior gradients for the navigation ROIs
	
	:param roi_masks: 			dict of ROI masks in fsaverage6
	:param position: 			posterior-anterior position of each vertex
	:param concreteness: 		concreteness index of each vertex
	:param significance_count: 	per-vertex count of significant participants
	:param num_bootstraps: 		number of bootstrap resamples for the confidence interval
	:return: dictionary of ROI name to its regression data
	"""
	gradients = {}
	for roi in roi_masks.keys():
		significant = numpy.logical_and(roi_masks[roi], significance_count > 2.5)
		position = position[significant]
		values = concreteness[significant]
		sorted_position, line, lower, upper, correlation = compute_slope(position, values, num_bootstraps)
		gradients[roi] = {'position': position, 'concreteness': values, 'sorted_position': sorted_position,
					  'line': line, 'lower': lower, 'upper': upper, 'correlation': correlation}
	return gradients


def significance_stars(p_value: float, correction_factor: int) -> str:
	"""
	Returns asterisks marking the significance of a Bonferroni-corrected p-value, one asterisk for
	each of the 1e-2, 1e-3, and 1e-4 levels that the corrected value falls below.

	:param p_value: 			uncorrected p-value
	:param correction_factor: 	number of comparisons to correct for
	:return: a string of asterisks
	"""
	stars = ''
	corrected = p_value * correction_factor
	for level in [1e-4, 1e-3, 1e-2]:
		if corrected < level:
			stars += '*'
	return stars


def compute_tuning_bias_nulls(roi_masks: dict, group_split_R2: numpy.ndarray, num_permutations: int = 10000) -> dict:
	"""
	Computes a null distribution of tuning bias for each ROI

	:param roi_masks: 			ROIs in fsaverage6
	:param group_split_R2: 		group-average split R2
	:param num_permutations: 	number of permutations
	:return: nulls for each ROI
	"""
	num_feature_spaces = group_split_R2.shape[0]
	nulls = {}
	for roi in roi_masks.keys():
		roi_mask = roi_masks[roi]
		null = numpy.zeros([num_permutations, num_feature_spaces])
		for permutation_index in range(num_permutations):
			permuted_mask = numpy.random.permutation(roi_mask)
			roi_mean = numpy.mean(group_split_R2[:, permuted_mask], axis = 1)
			other_mean = numpy.mean(group_split_R2[:, numpy.logical_not(permuted_mask)], axis = 1)
			null[permutation_index, :] = roi_mean - other_mean
		nulls[roi] = null
	return nulls
