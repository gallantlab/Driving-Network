from typing import List

import numpy
from sklearn.decomposition import PCA

from .DataInfo import NavIndicesForNPI


def get_weights_PCs(full_weights: List[numpy.ndarray]) -> List[numpy.ndarray]:
    out = []
    for w in full_weights:
        if w.shape[0] < 10:
            out.append(w)
        else:
            pca = PCA(n_components = 10)
            reduced = pca.fit_transform(w.transpose()).transpose()
            out.append(reduced)
    return out

def make_weight_PCA_objects(full_weights: List[numpy.ndarray]) -> list:
    out = []
    for w in full_weights:
        if w.shape[0] < 10:
            out.append(None)
        else:
            pca = PCA(n_components = 10)
            pca.fit(w.transpose())
            out.append(pca)
    return out


def get_navigation_score_for_one_clustering(clustering: numpy.ndarray, split_R2_scores: numpy.ndarray) -> numpy.ndarray:
    """
    clustering: a clustering solution of size N
    split_R2_scores: a [38 x N] model score matrix
    """
    clusters = numpy.max(numpy.unique(clustering)) + 1
    out = numpy.zeros_like(clustering, dtype = float)
    vals = []
    for cluster in range(1, clusters):
        navPerf = numpy.nanmean(split_R2_scores[:, clustering == cluster][NavIndicesForNPI, :])
        vals.append((navPerf, cluster))
    vals.sort(key = lambda x: -1 * x[0])
    out[clustering == vals[0][1]] = 1.0
    return out