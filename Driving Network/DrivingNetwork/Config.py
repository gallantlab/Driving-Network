import os

BASE_DIRECTORY = None

if BASE_DIRECTORY is not None:
    DATA_DIRECTORY = os.path.join(os.path.abspath(BASE_DIRECTORY), 'Data')
    FEATURES_DIRECTORY = os.path.join(os.path.abspath(BASE_DIRECTORY), 'Features')
    MODELS_DIRECTORY = os.path.join(os.path.abspath(BASE_DIRECTORY), 'Models')
    CLUSTERING_BOOTSTRAPS_DIRECTORY = os.path.join(os.path.abspath(BASE_DIRECTORY), 'Clustering')
    MAPPERS_DIRECTORY = os.path.join(os.path.abspath(BASE_DIRECTORY), 'Projection Matrices')
    MAPPER_FILE_PATH_TEMPLATE = os.path.join(MAPPERS_DIRECTORY, '{} {}.dat')

    DATA_PATH_TEMPLATE = os.path.join(DATA_DIRECTORY, '{}.hdf')
    FEATURES_PATH_TEMPLATE = os.path.join(FEATURES_DIRECTORY, '{}.hdf')

    MODEL_PATH_TEMPLATE = os.path.join(MODELS_DIRECTORY, '{}.dat')
    BOOTSTRAP_CLUSTERING_PATH_TEMPLATE = os.path.join(CLUSTERING_BOOTSTRAPS_DIRECTORY, '{}', 'clustering {}.npz')
    BOOTSTRAP_CLUSTERING_INFO_TEMPLATE = os.path.join(CLUSTERING_BOOTSTRAPS_DIRECTORY, '{}', 'clustering {} info.npz')
    
    NAVIGATION_ROI_OVERLAY_PATH = 'navigation overlay.svg'

    for path in [DATA_DIRECTORY, FEATURES_DIRECTORY, MODELS_DIRECTORY,
				 CLUSTERING_BOOTSTRAPS_DIRECTORY, MAPPERS_DIRECTORY]:
        if not os.path.exists(path):
            os.makedirs(path)
else:
    print('BASE_DIRECTORY not set; default paths will not work.')

NUM_CLUSTERING_BOOTSTRAPS = 200