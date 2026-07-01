# Plot tuning bias profiles for the navigation ROIs

from DrivingNetwork import IO, utils, Plotting

# load group split R2 and significance
results = IO.load_all_results()
converters = IO.load_individual_to_fsaverage_projections()
group_split_R2 = IO.load_group_split_R2(results, converters)
significance_count = IO.load_group_significance_count(results, converters)

# zero out non-significant vertices
group_split_R2[:, significance_count <= 2.5] = 0

# load navigation and comparison ROI masks
nav_rois, other_rois, _ = utils.load_roi_masks()
roi_masks = dict(nav_rois)
roi_masks.update({roi: other_rois[roi] for roi in ['V1', 'M1H']})

# null bias distribution for each ROI
null_biases = utils.compute_tuning_bias_nulls(roi_masks, group_split_R2)

# sunburst plots relative to the rest of cortex
Plotting.make_tuning_bias_figure(roi_masks, group_split_R2, null_biases,
								 'Positive tuning bias profiles')
Plotting.make_tuning_bias_figure(roi_masks, group_split_R2, null_biases,
								 'Negative tuning bias profiles', negative = True)
