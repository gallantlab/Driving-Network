from .Config import DATA_PATH_TEMPLATE, FEATURES_PATH_TEMPLATE, MAPPER_FILE_PATH_TEMPLATE, MODEL_PATH_TEMPLATE
from .DataInfo import Participants, Transform, FeatureSpaces
from .VM import Results

from typing import Dict, List

import numpy
from scipy import sparse
from .modelconn import fsaverage
import tables
import h5py

def load_brain_data(participant: Participants) -> [numpy.ndarray, numpy.ndarray]:
    """
    Loads brain data for one participant

    :param participant: 	participant to load data for
    :return: Tuple<train data, test data>
    """
    datafile = tables.open_file(DATA_PATH_TEMPLATE.format(participant.name))
    return datafile.root.train.read(), datafile.root.test.read()


def load_features(participant: Participants) -> [List[numpy.ndarray], List[numpy.ndarray]]:
    """
    Loads features for one participant

    :param participant: 	participant to load data for
    :return: train and test features as List<numpy array> in order of FeatureSpaces
    """
    datafile = tables.open_file(FEATURES_PATH_TEMPLATE.format(participant.name))
    train = [datafile.root.train.__getattr__(featureSpace) for featureSpace in FeatureSpaces]
    test = [datafile.root.test.__getattr__(featureSpace) for featureSpace in FeatureSpaces]
    return train, test


def load_VM_results(participant: Participants) -> Results:
    """
    Loads results from VM for a participant from location specified in configs

    :param participant:
    :return:
    """
    return Results.load(MODEL_PATH_TEMPLATE.format(participant.name))


def save_VM_results(results: Results) -> None:
    """
    Saves VM results for a participant to the location specified in the configs

    :param results:
    :return:
    """
    results.save(MODEL_PATH_TEMPLATE.format(results.participant.name))


def load_sparse_matrix_from_hdf(HDF_path: str, matrix_name: str) -> sparse.csr_matrix:
    """
    Loads a CSR sparse matrix from a HDF file

    :param HDF_path: 	HDF file path
    :param matrix_name: name of matrix in the HDF file
    :return:
    """
    HDF = h5py.File(HDF_path, 'r')
    matrix_data = numpy.array(HDF[matrix_name + '_data'])
    matrix_indices = numpy.array(HDF[matrix_name + '_indices'])
    matrix_index_pointers = numpy.array(HDF[matrix_name + '_indptr'])
    matrix_shape = numpy.array(HDF[matrix_name + '_shape'])
    return sparse.csr_matrix((matrix_data, matrix_indices, matrix_index_pointers), matrix_shape)


def load_individual_to_fsaverage_projections() -> Dict[Participants, fsaverage.Converter]:
    """
    Loads projections from subjects to fsaverage from location specified in config file.
    Returns are fsaverage Converter objects

    :return: a converter object for each participant
    """
    converters = {}

    for participant in Participants:
        mapper_path = MAPPER_FILE_PATH_TEMPLATE.format(participant.name)
        left = load_sparse_matrix_from_hdf(mapper_path, 'fsaverage-Left')
        right = load_sparse_matrix_from_hdf(mapper_path, 'fsaverage-Right')
        converter = fsaverage.Converter((left, right))
        converters[participant] = converter

    return converters


def load_fsaverage_upscaler(fsaverage_projections: Dict[str, Dict[str, numpy.ndarray]] = None) -> fsaverage.FSAverageUpscaler:
    """
    Loads just the upscaler for fsaverage

    :param fsaverage_projections:	pre-loaded projection matrices
    :return: fsaverage upscaler object
    """
    if fsaverage_projections is None:
        fsaverage_projections = {}
        file_path = MAPPER_FILE_PATH_TEMPLATE.format('fsaverage')
        for surface in ['fsaverage5', 'fsaverage6']:
            fsaverage_projections[surface] = {}
            for hemisphere in ['lh', 'rh']:
                fsaverage_projections[surface][hemisphere] = load_sparse_matrix_from_hdf(file_path,
																						 '{}-{}'.format(surface,
																										hemisphere))
    upscaler = fsaverage.FSAverageUpscaler(
            (fsaverage_projections['fsaverage5']['lh'], fsaverage_projections['fsaverage5']['rh']),
            (fsaverage_projections['fsaverage6']['lh'], fsaverage_projections['fsaverage6']['rh'])
            )
    return upscaler


def load_voxel_to_pixel_mappers() -> Dict[Participants, Dict]:
    """
    Loads matrices that projects from voxels to the pixels on individual participant flatmaps

    :return:	a projection matrix and a binary mask for the flatmap images for each participant
    """
    out = {}
    for participant in Participants:
        mapper_path = MAPPER_FILE_PATH_TEMPLATE.format(participant.name)
        pixel_mapper = load_sparse_matrix_from_hdf(mapper_path, 'voxel-to-flat-pixel')
        with h5py.File(mapper_path) as mapper_file:
            pixel_mask = numpy.array(mapper_file['pixel-mask'])
            out[participant] = {'pixel_mapper': pixel_mapper, 'pixel_mask': pixel_mask}
    return out


def load_all_results() -> Dict[Participants, Results]:
    """
    Loads the fit models for every participant.

    :return:	dictionary of participant to Results
    """
    return {participant: load_VM_results(participant) for participant in Participants}


def load_group_split_R2(results: Dict[Participants, Results] = None,
						converters: Dict[Participants, fsaverage.Converter] = None) -> numpy.ndarray:
    """
    Loads the group-average split R2 in fsaverage6

    :param results:		preloaded results per participant, if it exists
    :param converters:	preloaded fsaverage converters, if it exists
    :return: group-average split R2, shape [num feature spaces, num vertices]
    """
    from . import utils
    if results is None:
        results = load_all_results()
    return numpy.clip(numpy.nan_to_num(utils.to_group_average(
        {participant: results[participant].split_R2 for participant in Participants}, converters)), 0, 1)


def load_group_significance_count(results: Dict[Participants, Results] = None,
								  converters: Dict[Participants, fsaverage.Converter] = None) -> numpy.ndarray:
    """
    Get number of significant participants at each fsaverage6 vertex

    :param results:		preloaded results per participant, if any
    :param converters:	preloaded fsaverage converters, if any
    :return: significant count for each vertex
    """
    if results is None:
        results = load_all_results()
    if converters is None:
        converters = load_individual_to_fsaverage_projections()
    count = numpy.zeros(int(fsaverage.fsaverageResolution.fsaverage6) * 2)
    for participant in Participants:
        count += converters[participant].to_fsaverage(
            results[participant].significance.astype(float), fsaverage.fsaverageResolution.fsaverage6)
    return count