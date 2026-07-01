from typing import Dict, List, Optional

import numpy
from plotly import colors as PlotColors
from plotly import graph_objs as plots
from plotly.subplots import make_subplots
import cortex

from .DataInfo import FeatureSpaces, ModelCategories, ModelCategoryIndices, ModelCategoryMembers, ROIs
from . import utils
from .modelconn.fsaverage import FSAverageUpscaler

model_colors = {}
for i, feature_space in enumerate(FeatureSpaces):
	model_colors[feature_space] = PlotColors.sample_colorscale(
		PlotColors.cyclical.HSV, samplepoints = i / len(FeatureSpaces))[0]

model_category_colors = {}
for i, category in enumerate(ModelCategories):
	start, end = ModelCategoryIndices[category]
	model_category_colors[category] = PlotColors.sample_colorscale(
		PlotColors.cyclical.HSV, samplepoints = ((end - start) / 2.0 + start) / len(FeatureSpaces))[0]

NUM_NAV_ROIS = 11

def make_sunburst_parameters(values: numpy.ndarray, name: str, min_model_percentage: float = 0,
							 nulls: Optional[numpy.ndarray] = None, null_percentile: float = 95,
							 show_remainder: bool = False,
							 model_colors: Dict[str, str] = model_colors,
							 model_category_colors: Dict[str, str] = model_category_colors,
							 model_names: List[str] = FeatureSpaces) -> tuple:
	"""
	Makes parameters for a plotly sunburst plot by feature space. Optionally,
	if nulls are given, will zero out values that fall below the nth percentile of the null.

	:param values: 					data to plot; a 1d length-38 vector
	:param name: 					name to put at the center of the plot
	:param min_model_percentage: 	minimum percentage of total for a model to appear individually
	:param nulls: 					null values, if available
	:param null_percentile: 		percentile for null to consider values significant
	:param show_remainder: 			show the sum of values below min_model_percentage
	:param model_colors: 			colors for the models
	:param model_category_colors: 	colors for the model categories
	:param model_names: 			labels to use for feature spaces
	:return: labels, parents, values, and colors for use in plotly.graph_objects.Sunburst
	"""
	name = name.replace(' ', '<br>')

	plot_values = values.copy()
	if nulls is not None:
		null_thresholds = numpy.nanpercentile(nulls, null_percentile, axis = 0)
		plot_values[plot_values < null_thresholds] = 0
	plot_values = numpy.clip(plot_values, 0, 1)

	minimum_threshold = min_model_percentage * numpy.sum(plot_values)

	sunburst_labels = [name]
	sunburst_parents = ['']
	sunburst_values = [numpy.sum(plot_values)]
	sunburst_colors = ['rgba(0, 0, 0, 0)']

	for category in ModelCategories:
		category_start, category_end = ModelCategoryIndices[category]
		category_total = numpy.sum(plot_values[category_start:category_end])
		if category_total > 0:
			sunburst_labels.append(category.replace(' ', '<br>'))
			sunburst_parents.append(name)
			sunburst_values.append(category_total)
			if model_category_colors is not None:
				sunburst_colors.append(model_category_colors[category])
			else:
				mid_model_index = int(len(ModelCategoryMembers[category]) / 2)
				sunburst_colors.append(model_colors[ModelCategoryMembers[category][mid_model_index]])
			for feature_space in ModelCategoryMembers[category]:
				feature_index = utils.get_index(feature_space, FeatureSpaces)
				feature_value = plot_values[feature_index]
				if feature_value > minimum_threshold:
					sunburst_parents.append(category.replace(' ', '<br>'))
					sunburst_values.append(feature_value)
					sunburst_labels.append(model_names[feature_index])
					sunburst_colors.append(model_colors[feature_space])
					category_total -= feature_value
			if show_remainder and category_total > 0:
				sunburst_parents.append(category)
				sunburst_values.append(category_total)
				sunburst_labels.append('{} Remainder'.format(category).replace(' ', '<br>'))
				sunburst_colors.append('rgb(128,128,128)')

	return sunburst_labels, sunburst_parents, sunburst_values, sunburst_colors


def _make_roi_contour(reduced: numpy.ndarray, mask: numpy.ndarray, color: str, name: str, contour_start: float,
					  bin_size: float, histogram_normalization: str, dashed: bool) -> plots.Histogram2dContour:
	"""
	Makes a single 2D contour for an ROI's vertices on the UMAP embedding.

	:param reduced: 				2D UMAP embedding of every vertex
	:param mask: 					fsaverage6 boolean mask for the ROI
	:param color: 					line color
	:param name: 					ROI name
	:param contour_start: 			contour level at which to start drawing
	:param bin_size: 				bin size for the contour histogram
	:param histogram_normalization: plotly histogram normalization mode
	:param dashed: 					whether to draw the contour dashed
	:return: a plotly Histogram2dContour trace
	"""
	line = {'color': color, 'width': 4, 'smoothing': 1.3}
	if dashed:
		line['dash'] = 'dash'
	return plots.Histogram2dContour(x = reduced[mask, 0], y = reduced[mask, 1], line = line,
									hoverinfo = 'name', name = name, opacity = 0.75,
									contours = {'coloring': 'none', 'start': contour_start, 'end': 0.9},
									histnorm = histogram_normalization,
									ybins = {'size': bin_size, 'start': -6, 'end': 7},
									xbins = {'size': bin_size, 'start': -9, 'end': 5},
									ncontours = 2)


def make_umap_roi_figure(reduced: numpy.ndarray, navigation_masks: dict, comparison_masks: dict,
						 contour_start: float = 0.025, bin_size: float = 0.5) -> plots.Figure:
	"""
	Makes the UMAP plot of cortical vertices, and draws contour lines for the navigation ROIs,
	along with some other visual and motor ROIs for comparison
	
	:param reduced: 			2D UMAP embedding of every vertex, shape [num vertices, 2]
	:param navigation_masks: 	navigation ROI masks
	:param comparison_masks: 	comparison ROI masks
	:param contour_start: 		contour level at which to start drawing
	:param bin_size: 			bin size for the contour histograms
	:return: a plotly Figure
	"""
	histogram_normalization = 'probability'
	total_roi_count = len(navigation_masks) + len(comparison_masks)
	roi_union = numpy.zeros(reduced.shape[0], dtype = bool)
	for masks in [navigation_masks, comparison_masks]:
		for roi in masks:
			roi_union = numpy.logical_or(roi_union, masks[roi])

	plot_data = [plots.Scattergl(x = reduced[numpy.logical_not(roi_union), 0],
								 y = reduced[numpy.logical_not(roi_union), 1], mode = 'markers', text = 'Non-ROI',
								 marker = {'color': 'rgb(128, 128, 128)', 'size': 5,
										   'line': {'color': 'rgba(0, 0, 0, 0)'}, 'opacity': 0.025}, name = 'Non-ROI')]

	for i, roi in enumerate(navigation_masks.keys()):
		color = PlotColors.sample_colorscale(PlotColors.cyclical.HSV, samplepoints = i / total_roi_count)[0]
		plot_data.append(plots.Scattergl(x = reduced[navigation_masks[roi], 0], y = reduced[navigation_masks[roi], 1],
										 mode = 'markers', text = roi,
										 marker = {'color': color, 'size': 8, 'line': {'color': 'rgba(0, 0, 0, 0)'},
												   'opacity': 0.1}, name = roi, showlegend = False))

	for i, roi in enumerate(navigation_masks.keys()):
		color = PlotColors.sample_colorscale(PlotColors.cyclical.HSV, samplepoints = i / total_roi_count)[0]
		plot_data.append(_make_roi_contour(reduced, navigation_masks[roi], color, roi, contour_start, bin_size,
										   histogram_normalization, False))

	for i, roi in enumerate(comparison_masks.keys()):
		color = PlotColors.sample_colorscale(PlotColors.cyclical.HSV,
											 samplepoints = (i + len(navigation_masks)) / total_roi_count)[0]
		plot_data.append(_make_roi_contour(reduced, comparison_masks[roi], color, roi, contour_start, bin_size,
										   histogram_normalization, True))

	figure = plots.Figure(data = plot_data)
	figure.update_layout(width = 1600, height = 900, yaxis = {'scaleanchor': 'x'}, font = {'size': 24},
						 margin = {'l': 20, 'r': 20, 't': 64, 'b': 5},
						 title = 'Functional distribution of all vertices'.format(contour_start, bin_size))
	return figure


def make_umap_flatmap(reduced: numpy.ndarray, performance: numpy.ndarray, significance_count: numpy.ndarray,
					  upscaler: FSAverageUpscaler, opacity_ceiling: float = 0.25) -> cortex.Vertex2D:
	"""
	Map UMAP functional embedding of vertices to the cortical surface witha 2D colormap

	:param reduced: 			2D UMAP embedding of every vertex, shape [num vertices, 2]
	:param performance: 		group-average prediction performance in fsaverage6
	:param significance_count: 	per-vertex count of significant participants in fsaverage6
	:param upscaler: 			fsaverage upscaler
	:param opacity_ceiling: 	max value for full opacity
	:return: pycortex dataset
	"""
	transparency = performance.copy()
	transparency[significance_count < 3] = 0
	transparency = numpy.nan_to_num(numpy.sqrt(transparency))
	transparency = numpy.clip(upscaler.to_fsaverage(transparency), 0, opacity_ceiling) / opacity_ceiling

	percentile_low = numpy.percentile(reduced, 1, axis = 0)
	percentile_high = numpy.percentile(reduced, 99, axis = 0)
	dimension1 = numpy.clip(reduced[:, 0], percentile_low[0], percentile_high[0])
	dimension2 = numpy.clip(reduced[:, 1], percentile_low[1], percentile_high[1])

	vertex = cortex.Vertex2D(upscaler.to_fsaverage(dimension1), upscaler.to_fsaverage(dimension2), 'fsaverage',
							 cmap = 'spatial', vmin = percentile_low[0], vmax = percentile_high[0],
							 vmin2 = percentile_low[1], vmax2 = percentile_high[1])
	return vertex.blend_curvature(transparency, contrast = 0.1, brightness = 0.1)


def _concreteness_roi_color(index: int) -> str:
	"""
	Helper for determining colors in the concreteness plots

	:param index: 				position of the ROI in the ordered list
	:return: an rgb color string
	"""
	if index < NUM_NAV_ROIS:
		return PlotColors.sample_colorscale(PlotColors.cyclical.Phase, index / float(NUM_NAV_ROIS))[0]
	return PlotColors.sample_colorscale(PlotColors.cyclical.mrybm, (index - NUM_NAV_ROIS) / 2)[0]


def _calc_plot_limits(x_values: numpy.ndarray, y_values: numpy.ndarray) -> tuple:
	"""
	Given values to plot, determine the limits for the plot such that it is a square with 1:1 X:Y scaling

	:param x_values: 	values for the horizontal axis
	:param y_values: 	values for the vertical axis
	:return: rescaled x, rescaled y, original x range, original y range, and the scale factor
	"""
	x_low = numpy.min(x_values)
	x_high = numpy.max(x_values)
	y_low = numpy.min(y_values)
	y_high = numpy.max(y_values)
	x_range = (x_high - x_low)
	y_range = (y_high - y_low)
	rescaler = x_range if x_range > y_range else y_range
	x_out = (x_values - x_low) / rescaler
	y_out = (y_values - y_low) / rescaler
	return x_out, y_out, (x_low, x_high), (y_low, y_high), rescaler

def make_concreteness_figure_individual_rois(gradient_slopes: dict) -> plots.Figure:
	"""
	Draws the concreteness-against-position gradient for each navigation ROI in its own square
	subplot, rescaling each ROI to fill its subplot.

	:param gradient_slopes: 	A/P and concreteness values for ROI vertices, regression line, and confidence intervals
	:return: a plotly Figure
	"""
	figure = make_subplots(rows = 4, cols = 3, subplot_titles = ROIs + [''],
						   vertical_spacing = 0.05, horizontal_spacing = 0.05)
	for i, roi in enumerate(ROIs):
		row = int(i / 3) + 1
		column = i % 3 + 1
		color = _concreteness_roi_color(i)
		gradient_slop = gradient_slopes[roi]

		position, concreteness_values, position_range, concreteness_range, rescaler = \
			_calc_plot_limits(gradient_slop['position'], gradient_slop['concreteness'])
		x_offset = 0.5 - (numpy.max(position) / 2.0)
		y_offset = 0.5 - (numpy.max(concreteness_values) / 2.0)
		position_span = position_range[1] - position_range[0]
		concreteness_span = concreteness_range[1] - concreteness_range[0]
		position += x_offset
		concreteness_values += y_offset

		figure.add_trace(plots.Scatter(x = position, y = concreteness_values, mode = 'markers',
									   marker = {'color': 'rgba' + color[3:-1] + ',0.1)'}, showlegend = False),
						 row = row, col = column)

		confidence_interval = numpy.array(gradient_slop['lower'].tolist() + gradient_slop['upper'].tolist()[::-1])
		confidence_interval = (confidence_interval - concreteness_range[0]) / rescaler + y_offset
		sorted_position = numpy.sort(position).tolist()
		figure.add_trace(plots.Scatter(x = sorted_position + sorted_position[::-1], y = confidence_interval.tolist(),
									   fillcolor = color, showlegend = False, fill = 'toself', opacity = 0.25,
									   line = {'color': 'rgba(0,0,0,0)'}), row = row, col = column)
		line = (gradient_slop['line'] - concreteness_range[0]) / rescaler + y_offset
		figure.add_trace(plots.Scatter(x = sorted_position, y = line, showlegend = False, line = {'color': color}),
						 row = row, col = column)
		figure.update_xaxes(tickmode = 'array', tickvals = [0, 1],
							ticktext = ['{:1.2f}'.format(value + x_offset * position_span) for value in position_range],
							row = row, col = column)
		figure.update_yaxes(tickmode = 'array', tickvals = [0, 1],
							ticktext = ['{:1.2f}'.format(value + y_offset * concreteness_span) for value in
										concreteness_range],
							row = row, col = column)

	figure.update_layout(margin = {'l': 0, 'r': 0, 'b': 0, 't': 0}, width = 600, height = 800, font_family = 'Arial')
	for column in range(3):
		figure.update_yaxes(scaleanchor = 'x{}'.format(column + 1), col = column + 1, range = [-0.1, 1.1])
		figure.update_xaxes(scaleanchor = 'x1', range = [-0.1, 1.1], col = column + 1)
	return figure


def make_concreteness_figure_all_rois(posterior_anterior_scale: numpy.ndarray, concreteness: numpy.ndarray,
									  navigation_union: numpy.ndarray, significance_count: numpy.ndarray,
									  num_bootstraps: int = 5000) -> plots.Figure:
	"""
	Draws the concreteness-against-position gradient across for vertices in all navigation ROIs

	:param posterior_anterior_scale: 	posterior-anterior position of each vertex
	:param concreteness: 				concreteness index of each vertex
	:param navigation_union: 			fsaverage6 mask for all navigation vertices
	:param significance_count: 			per-vertex count of significant participants
	:param num_bootstraps: 				number of bootstrap resamples for the confidence interval
	:return: a plotly Figure
	"""
	significant = numpy.logical_and(navigation_union, significance_count > 2.5)
	all_position, all_concreteness = posterior_anterior_scale[significant], concreteness[significant]
	sections = {'posterior': (all_position[all_position < 0], all_concreteness[all_position < 0]),
				'prefrontal': (all_position[all_position > 0], all_concreteness[all_position > 0]),
				'all': (all_position, all_concreteness)}
	section_colors = PlotColors.qualitative.D3

	plot_data = [plots.Scatter(x = sections['posterior'][0], y = sections['posterior'][1], mode = 'markers',
							   marker = {'color': 'rgba(31,119,180,0.1)'}, showlegend = False),
				 plots.Scatter(x = sections['prefrontal'][0], y = sections['prefrontal'][1], mode = 'markers',
							   marker = {'color': 'rgba(255,127,14,0.1)'}, showlegend = False)]

	for i, section in enumerate(['posterior', 'prefrontal', 'all']):
		position, values = sections[section]
		sorted_position, line, lower, upper, correlation = utils.compute_slope(position, values, num_bootstraps)
		stars = utils.significance_stars(correlation[1], 3)
		color = section_colors[i]
		sorted_position = sorted_position.tolist()
		plot_data.append(
			plots.Scatter(x = sorted_position + sorted_position[::-1], y = lower.tolist() + upper.tolist()[::-1],
						  fillcolor = color, showlegend = False, fill = 'toself', opacity = 0.25,
						  line = {'color': 'rgba(0,0,0,0)'}))
		plot_data.append(plots.Scatter(x = sorted_position, y = line, line = {'color': color},
									   name = '{} r = {:.4f}{} p = {:.2e}'.format(section, correlation[0], stars,
																				  correlation[1])))

	return plots.Figure(data = plot_data,
						layout = plots.Layout(title = 'Posterior-anterior overall', width = 800, height = 400,
											  yaxis = {'range': [-0.75, 1], 'title': 'Concreteness', 'ticks': 'inside',
													   'mirror': True},
											  xaxis = {'range': [-0.75, 1], 'title': 'Posterior-anterior',
													   'ticks': 'inside',
													   'mirror': True, 'scaleanchor': 'y', 'scaleratio': 2}))


def make_tuning_bias_figure(roi_masks: dict, group_split_R2: numpy.ndarray, null_biases: dict,
							title: str, negative: bool = False) -> plots.Figure:
	"""
	Builds sunburst plots showing each ROI's tuning bias profiles

	:param roi_masks: 		ROI masks in fsaverage6
	:param group_split_R2: 	group-average split R2
	:param null_biases: 	null biases for each ROI
	:param title: 			figure title
	:param negative: 		whether to show the under-represented feature spaces
	:return: a plotly Figure
	"""
	sign = -1 if negative else 1
	figure = make_subplots(rows = 4, cols = 4, specs = [[{'type': 'domain'}] * 4] * 4,
						   horizontal_spacing = 0.01, vertical_spacing = 0.01)
	for plot_index, roi in enumerate(roi_masks.keys()):
		roi_mean = numpy.mean(group_split_R2[:, roi_masks[roi]], axis = 1)
		other_mean = numpy.mean(group_split_R2[:, numpy.logical_not(roi_masks[roi])], axis = 1)
		labels, parents, values, colors = make_sunburst_parameters(sign * (roi_mean - other_mean), roi,
																   nulls = sign * null_biases[roi])
		figure.add_trace(plots.Sunburst(labels = labels, parents = parents, values = values,
										marker = {'colors': colors}, branchvalues = 'total'),
						 row = int(plot_index / 4) + 1, col = plot_index % 4 + 1)
	figure.update_layout(width = 1200, height = 900, title = title, margin = {'t': 48, 'l': 0, 'r': 0, 'b': 0})
	figure.update_traces(sort = False, selector = {'type': 'sunburst'})
	return figure
