""" Defines linkage class to generate clustering solutions. """

import math
from enum import Enum
from typing import Optional, List

import networkx as nx
import numpy
import numpy as np

try:
    import fastcluster
except:
    fastcluster = None

from scipy.cluster import hierarchy
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import pdist, squareform

from .fsaverage import Converter, fsaverageResolution

try:
    import Pairwise
except:
    Pairwise = None


class Metrics():
    """
    Enum of possible distance metrics
    """
    BrayCurtis = 'braycurtis'
    Canberra = 'canberra'
    Chebyshev = 'chebyshev'
    CityBlock = 'cityblock'
    Correlation = 'correlation'
    Cosine = 'cosine'
    Dice = 'dice'
    Euclidean = 'euclidean'
    Hamming = 'hamming',
    Jaccard = 'jaccard'
    JensonShannon = 'jensenshannon'
    Kulsinski = 'kulsinski'
    Mahalanobis = 'mahalanobis'
    matching = 'matching'
    Minkowski = 'minkowski'
    RogersTanimoto = 'rogerstanimoto'
    RussellRao = 'russellrao'
    StandardizedEuclidean = 'seuclidean'
    SokalMichener = 'sokalmichener'
    SokalSneath = 'sokalsneath'
    SquareEuclidean = 'sqeuclidean'
    Yule = 'yule'
    RandomForest = 'RandomForest'


class LinkageMethods(Enum):
    """
    Enum of possible linkage methods in sklearn
    """
    Single = 'single'
    Complete = 'complete'
    Average = 'average'
    Weighted = 'weighted'
    Centroid = 'centroid'
    Median = 'median'
    Ward = 'ward'


def get_graph(corrmat: np.ndarray, threshold: float = 0.4) -> nx.Graph:
    """ Returns correlation matrix as a networkx graph.

    Parameters
    ----------
    corrmat : array of shape (n_clusters, n_clusters)
    threshold : float

    Returns
    -------
    G : networkx.Graph
    """
    n_nodes = corrmat.shape[0]
    G = nx.Graph()
    for i in range(n_nodes):
        for j in range(n_nodes):
            weight = corrmat[i,j]
            if i==j or math.isnan(weight):
                continue
            if weight > threshold:
                G.add_edge(i+1, j+1, weight=weight)
    return G


def get_dists(X: np.ndarray, as_mat: bool, metric: str = "euclidean") -> np.ndarray:
    """ Returns pairwise distances between X.

    Parameters
    ---------
    X : array of shape (n_targets, n_features)
    as_mat : bool
        If true, return the data as a pairwise matrix.
        Else return a list of pairwise distances.
    metric : str
        The distance metric to use.
        See scipy.spatial.distance for all options.

    Returns
    -------
    distances : an array of shape (n_targets, n_targets) or a list
    """
    if (Pairwise is not None) and ( metric in [Metrics.Correlation, Metrics.Euclidean, Metrics.RandomForest]):
        n = X.shape[0]
        dists = numpy.zeros(int(n * (n - 1) / 2))
        if metric == Metrics.Correlation:
            Pairwise.GetPairwiseCorrelationDistance(X, dists)
        elif metric == Metrics.Euclidean:
            Pairwise.GetPairwiseEuclideanDistance(X, dists)
        elif metric == Metrics.RandomForest:
            Pairwise.GetPairwiseRandomForestDistance(X, dists)
    else:
        dists = pdist(X, metric = metric).astype("float32")
    if not as_mat:
        return dists
    else:
        return squareform(dists)


def match_clusters(clustering_A: np.ndarray, clustering_B: np.ndarray,
                   weights_A: Optional[np.ndarray] = None,
                   weights_B: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Get the best matches in clustering B to those in clustering A based on spatial overlap

    Parameters
    ----------
    clustering_A
        The reference clustering labels
    clustering_B
        The clustering labels to match to the reference
    weights_A
        If given, cluster labels are matched based on weight similarities
    weights_B
        If given, cluster labels are matched based on weight similarities

    Returns
    -------
    new_labels
        new labels for B that best matches each cluster in B to a cluster in A
    """
    reference_cluster_labels, similarities = calculate_cluster_similarities(clustering_A, clustering_B, weights_A, weights_B)

    # maximize matching
    assignments = linear_sum_assignment(similarities, maximize = True)[1]

    # get colors that match assignment
    new_labels = np.array(reference_cluster_labels)[assignments]
    return new_labels.astype(int)


def calculate_cluster_similarities(clustering_A: np.ndarray, clustering_B: np.ndarray,
                                   weights_A: Optional[np.ndarray] = None,
                                   weights_B: Optional[np.ndarray] = None,
                                   ignore_zeros: bool = True) -> tuple:
    """
    Calculates the similarities between the clusters in the two clusterings
    based on spatial overlap

    Parameters
    ----------
    clustering_A
        The reference clustering labels
    clustering_B
        The clustering labels to match to the reference
    weights_A
        If given, cluster labels are matched based on weight similarities
    weights_B
        If given, cluster labels are matched based on weight similarities
    ignore_zeros
        If given, don't compute dice for the cluster with label 0

    Returns
    -------
    clustering_A_labels
        labellings used in the reference cluster
    similarities
        The similarities between pairwise clusters from the two clusterings

    """
    clustering_A_labels = np.unique(clustering_A)
    clustering_B_labels = np.unique(clustering_B)
    if (clustering_A.shape != clustering_B.shape):
        raise ValueError("Clusterings must be in same space")
    if ignore_zeros:
        # numpy.unique returns sorted labels, so 0 should be the first
        if 0 in clustering_A_labels:
            clustering_A_labels = clustering_A_labels[1:]
        if 0 in clustering_B_labels:
            clustering_B_labels = clustering_B_labels[1:]

    num_clusters_A = len(clustering_A_labels)
    num_clusters_B = len(clustering_B_labels)

    weights = (weights_A is not None) and (weights_B is not None)
    # calculate similarities

    similarities = np.zeros((num_clusters_A, num_clusters_B))
    rowIndices = []		# cluster numberings from clustering A
    columnIndices = [] 	# cluster numberings from clustering B
    for i, B_cluster in enumerate(clustering_B_labels):
        cluster = (clustering_B == B_cluster)
        rowIndices.append(B_cluster)

        for j, A_cluster in enumerate(clustering_A_labels):
            ref_cluster = (clustering_A == A_cluster)
            columnIndices.append(A_cluster)

            # calculate dice coefficient
            union = np.sum(cluster) + np.sum(ref_cluster)
            intersect = np.sum(cluster * ref_cluster)
            dice = 2. * intersect / union

            similarities[j, i] = dice

            if (weights):
                # calculate weight correlation
                cor = np.corrcoef(weights_B[:, cluster].mean(axis = 1),
                                  weights_A[:, ref_cluster].mean(axis = 1))
                weight_cor = cor[0, 1]
                similarities[j, i] += weight_cor
    return similarities, rowIndices, columnIndices


def relabel_clusters(clustering: np.ndarray, new_labels: Optional[np.ndarray]) -> np.ndarray:
    """ Relabels clustering such that each cluster is replaced
    with the corresponding value in cluster_labels.

    Parameters
    ----------
    clustering :  array of shape (n_targets)

    Returns
    -------
    relabelled :  array of shape (n_targets)
    """
    if new_labels is None:
        return clustering

    relabelled = np.zeros(clustering.shape[0])
    for col_i, cluster in zip(new_labels, np.unique(clustering)):
        assert len(clustering == cluster) > 0
        relabelled[clustering == cluster] = col_i

    return relabelled


class Brainspace(Enum):
    Individual = 'indiv'
    fsaverage5 = 'fsavg5'
    fsaverage6 = 'fsavg6'
    fsaverage7 = 'fsavg7'
    fsaverage  = 'fsavg'


# noinspection PyBroadException
class Linkage(object):
    """ Stores and retreives linkage matrix
    that is used to produce clustering solutions.

    Parameters
    ----------
    linkage : str
        The base file name for the linkage matrix.
        ex. S01-10_english1000-avg_euc
            S03_BOLD_cor
    brainspace : int
        The brain space that the data is in.
        Options are {"indiv", "fsavg5", "fsavg6", "fsavg"}.
    Z : scipy.cluster.hierarchy.linkage
        Optional, the linkage matrix that represents
        the similarities between all targets.
    """

    @staticmethod
    def load(file_name: str) -> "Linkage":
        """
        Loads a saved linkage object
        """
        saved = np.load(file_name, allow_pickle = True)
        linkage = Linkage(str(saved.get('subject')), saved.get('transform'), Z = saved.get('Z'))
        linkage._num_voxels = saved.get('num_voxels')
        try:
            linkage.linkage_parameters = saved.get('params')
        except:	# params was not saved, ignore
            pass
        try:
            linkage.mask = saved.get('mask')
        except:	# mask not saved or is none
            pass
        return linkage


    def __init__(self, subject: str = None, transform: Optional[str] = None,
                 Z: Optional[np.ndarray] = None, brainspace = Brainspace.Individual,
                 fsaverage_converter: Optional[Converter] = None) -> None:
        self.Z: Optional[np.ndarray] = Z				# Z matrix that describes the clustering
        self.linkage_parameters = {}					# parameters that describe how the linkage was created
        self.subject = subject							# pycortex subject
        self.transform = transform						# optional, what pycortex transform is this under
        self.cluster_colors: Optional[List[int]] = None # cluster color assignments
        self._num_voxels = 0							# number of voxels/vertices used
        self.brainspace = brainspace					# what space is the clustering data performed in
        self.fsaverage_converter = fsaverage_converter	# object to move from subject space to fsaverage
        self.mask = None								# a mask for the data to back to volumetric space
        self.X = None									# data that the linkage was fit to


    def print_linkage_params(self) -> None:
        """ Prints the parameters used to construct the linkage matrix.
        Only works if params were saved with linkage matrix.
        """
        # loader = np.load(self.linkage_file, allow_pickle=True)
        # try:
        # 	print(load_npz_dict(loader.get("params")))
        # except:
        # 	print("No params saved")

        if len(self.linkage_parameters.keys()) > 0:
            print(self.linkage_parameters)
        else:
            print("No params saved")


    def get_linkage(self) -> np.ndarray:
        """ Returns the hierarchical clustering solution
        encoded as a linkage matrix.

        Returns
        -------
        Z : array of shape (n_targets-1, 4)
            The hierarchical clustering solution.
        """
        if self.Z is not None:
            # return stored linkage
            return self.Z
        else:
            raise ValueError("Linkage not yet computed")


    def unmask(self, data: numpy.ndarray) -> numpy.ndarray:
        """
        Unmask some data
        Parameters
        ----------
        data
            1 or 2 d array; if 2D, then the 1st dimension is the masked dimension
        Returns
        -------
            unmasked data
        """
        if self.mask is None:
            raise ValueError('No mask stored')

        if len(data.shape) == 1:
            out = np.zeros(self.mask.shape[0])
            out[self.mask] = data
        else:
            out = np.zeros([self.mask.shape[0], data.shape[1]])
            out[self.mask, :] = data
        return out


    def get_X(self, fsaverage: Optional[fsaverageResolution] = None) -> np.ndarray:
        """
        Get the data this was fit on, will unmask and optionally project to fsaverage
        Parameters
        ----------
        fsaverage
            fsaverage resolution to project to, if None, is subject native

        Returns
        -------

        """
        if (self.X is None):
            raise ValueError("Linkage not yet computed")

        out = self.X
        if (self.mask is not None):
            out = self.unmask(self.X)

        if fsaverage is not None:
            return np.nan_to_num(self.fsaverage_converter.to_fsaverage(out, fsaverage))

        return out


    def fit(self, X: np.ndarray, metric: [Metrics, str] = Metrics.Correlation,
            unmask: Optional[np.ndarray] = None,
            method: [LinkageMethods, str] = LinkageMethods.Ward) -> None:
        """
        Performs Ward's hierarchical clustering on X data
        and returns the resulting linkage matrix.

        Parameters
        ----------
        X : array of shape (n_targets, n_features)
            Data to cluster
        metric : str
            The distance metric to use.
        unmask:
            A binary mask for this data to go back to a space pycortex recognizes
        method:
            How to compute the distance between clusters?

        Returns
        -------
        Z : scipy.hierarchy.linkage
        """
        if type(metric) == Metrics:
            metric = str(metric.value)

        if type(method) == LinkageMethods:
            method = str(method.value)

        print(f" {metric} pdist on X: "
              f"n_targets={X.shape[0]}, n_features={X.shape[1]}")
        dists = get_dists(X, as_mat=False, metric = str(metric))
        dists = np.nan_to_num(dists)

        print(" constructing linkage")
        if fastcluster is not None:
            self.Z = fastcluster.linkage(dists, method = method)
        else:
            self.Z = hierarchy.linkage(dists, method = method,
                                     optimal_ordering=False)
        self._num_voxels = X.shape[0]
        self.mask = unmask
        self.X = X


    def save(self, file_name: str, params: dict = None) -> None:
        """ Saves the hierarchical clustering solution.

        Either uses the self.Z or generates a Z matrix from
        the given X data. Does not allow over-writing.

        Parameters
        ----------
        file_name : str
            file name to save to
        params : dict
            Dictionary of parameters used to generate data.
        """
        out = {'Z': self.Z,
               'subject': self.subject,
               'num_voxels': self._num_voxels}
        if params is not None:
            out['params'] = params
        if self.mask is not None:
            out['mask'] = self.mask
        if self.transform is not None:
            out['transform'] = self.transform

        np.savez(file_name, **out)


    def get_clustering(self, num_clusters: int, match_colors: bool = False, cluster_colors: List[int] = None,
                       fsaverage: Optional[fsaverageResolution] = None, unmask: bool = False) -> np.ndarray:
        """ Returns cluster assignment for each voxel or vertex.

        Parameters
        ----------
        num_clusters : int
            The number of clusters.
        match_colors : bool
            If true, assign clusters colors that minimizes changes in color
            as the number of networks increases. Else, do not reorder.
        cluster_colors : list of int
            Indicates which color to assign to each cluster.
        fsaverage :
            if not none, what fsaverage resolution to return the clustering in
            if none, returns in the same space the data was given in
        unmask:
            if unmask exists, unmask the data?

        Returns
        -------
        clustering : array of shape (n_targets)
            A cluster assignment for each target.
        """
        Z = self.get_linkage()
        # assignment for each voxel/vertex
        clustering = hierarchy.fcluster(Z, num_clusters, criterion= "maxclust")

        # maybe get default labels
        if match_colors:
            if cluster_colors is None:
                cluster_colors = self.get_cluster_colors(num_clusters)
            clustering = relabel_clusters(clustering, cluster_colors)

        out_data_type = numpy.uint8
        if num_clusters > 255:
            out_data_type = numpy.uint16
        elif num_clusters > 65535:	# when would this happen??
            out_data_type = numpy.uint32

        # unmask
        if (unmask or (fsaverage is not None)) and (self.mask is not None):
            unmasked = np.zeros_like(self.mask, dtype = clustering.dtype)
            unmasked[self.mask] = clustering

            # maybe project to different brainspace
            if fsaverage is not None:
                out = numpy.zeros(int(fsaverage) * 2)
                for i in range(num_clusters + 1):
                    native_cluster = unmasked == i
                    fsaverage_cluster = self.fsaverage_converter.to_fsaverage(native_cluster, fsaverage)
                    out[fsaverage_cluster > 0.5] = i
                return out.astype(out_data_type)
            else:
                return unmasked.astype(out_data_type)

        return clustering.astype(out_data_type)


    def get_clustering_avgs(self, num_clusters: int, X: np.ndarray = None, cluster_order=None) -> tuple:
        """ Returns the average data for each cluster.

        Parameters
        ----------
        num_clusters : int
            The number of clusters.
        X : array of shape (n_targets, n_features)
            The data to be averaged across clusters.
        cluster_order : list of int
            Optional, if given, return cluster averages in
            the given order. Defaults to starting at cluster == 1
            and proceeding until cluster == t.

        Returns
        -------
        avgs : array of shape (num_clusters, num_features)
            The average feature values across targets in each cluster.
        stds : array of shape (num_clusters, num_features)
            The standard deviation of feature values across targets in each cluster.
        """
        clustering = self.get_clustering(num_clusters, match_colors = False)
        if cluster_order is None:
            cluster_order = np.unique(clustering)

        if X is None:
            X = self.X

        if X.ndim == 1:
            avgs = np.zeros(num_clusters)
            stds = np.zeros(num_clusters)
            for i, t_i in enumerate(cluster_order):
                avgs[i] = np.mean(X[clustering==t_i])
                stds[i] = np.std(X[clustering==t_i])
        elif X.ndim == 2:
            avgs = np.zeros((num_clusters, X.shape[1]))
            stds = np.zeros((num_clusters, X.shape[1]))
            for i, t_i in enumerate(cluster_order):
                avgs[i] = np.mean(X[clustering==t_i,:], axis=0)
                stds[i] = np.std(X[clustering==t_i,:])
        else:
            raise ValueError('X number of dimensions not equal to 1 or 2')
        return avgs, stds


    def get_cluster_colors(self, num_clusters: int) -> list:
        """ Returns the color order that produces consistent colors
        between ts for the given t.

        Parameters
        ----------
        num_clusters : int
            The number of clusters.

        Returns
        -------
        self.cluster_colors[t] : list of int
            The cluster ordering.
        """
        if (self.cluster_colors is None) or (num_clusters not in self.cluster_colors):
            t_max = np.max((num_clusters, 30))

            colors = {2: (1, 2)}
            # maybe swap initial cluster order
            Z = self.get_linkage()
            fcluster = hierarchy.fcluster(Z, 2, criterion="maxclust")
            if sum(fcluster == 1) < sum(fcluster == 2):
                colors[2] = (2, 1)

            # find best order for range of ts
            for t_i in range(3, t_max+1):
                colors[t_i] = self._get_best_order(t_i, colors[t_i-1])
            self.cluster_colors = colors

        return self.cluster_colors[num_clusters]


    def _get_best_order(self, num_clusters: int, ref_colors: list) -> list:
        """ Matches clusterings between t-1 and t.

        Parameters
        ----------
        num_clusters : int
        ref_colors : list of int
            The cluster indices corresponding to the t-1 clustering.

        Returns
        -------
        best_order : list of int
            List of values ranging from 0 to t-1. Indicates the cluster
            order for t clusters that preserves the cluster assignments
            from t-1 clusters.
        """
        # get clustering of t networks
        fcluster = self.get_clustering(num_clusters, match_colors =False)
        n_targets = fcluster.shape[0]
        # Z = self.get_linkage()
        # n_targets = Z.shape[0] + 1
        # fcluster = hierarchy.fcluster(Z, t, criterion="maxclust").astype("float32")

        # reference is a clustering of t-1
        fcluster_ref = self.get_clustering(num_clusters - 1, cluster_colors=ref_colors)
        assert np.min(fcluster_ref) == 1

        # save best order
        best_colors = np.arange(1, num_clusters + 1)
        max_n_match = (fcluster_ref==fcluster).sum()
        for i in range(num_clusters):
            # for each index, add the new cluster value
            colors_i = list(ref_colors).copy()
            colors_i.insert(i, num_clusters)

            # reorder clustering
            fcluster_order = np.zeros(n_targets)
            for color, cluster in zip(colors_i, np.unique(fcluster)):
                fcluster_order[fcluster==cluster] = color
            assert np.max(fcluster_order) > np.max(fcluster_ref)

            # evaluate new ordering
            n_match = (fcluster_ref==fcluster_order).sum()
            # print(colors_i, n_match)
            if n_match > max_n_match:
                best_colors = colors_i
                max_n_match = n_match

        return best_colors


    def get_clustering_split(self, num_clusters: int, t_split: int, network_i: int = None) -> dict:
        """ Uses t to define networks and t_split to define ROIs.

        Parameters
        ----------
        num_clusters : int
            The number of networks to define.
        t_split : int
            The number of parcels to split t into.
        network_i : int
            Optional. If an int, returns the split for a specific
            network. Else, returns the dictionary.

        Returns
        -------
        cluster_splits : dict of tuples
            Each element indicates the min and max cluster index.
        """
        # smaller number of networks
        networks = self.get_clustering(num_clusters, match_colors =False)
        # .. are split into a larger number of areas
        parcels = self.get_clustering(t_split, match_colors =False)

        cluster_splits = {}
        cluster_t_rng = [1,1]
        for network in np.unique(networks):
            n_network = np.sum(networks==network)
            # print(n_network)

            n_v = 0
            p_i = cluster_t_rng[1]
            while n_v != n_network:
                n_v += np.sum(parcels==p_i)
                # print(p_i, n_v)
                p_i += 1

                if n_v == n_network:
                    break
                if n_v > n_network:
                    raise ValueError()

            cluster_t_rng = [cluster_t_rng[0], p_i]

            cluster_splits[network] = cluster_t_rng.copy()
            cluster_t_rng[0] = cluster_t_rng[1]
            cluster_t_rng[1] = cluster_t_rng[0]

        if network_i is None:
            return cluster_splits
        else:
            return cluster_splits[network_i]


    def get_sub_clustering(self, network_i: int, num_networks: int, t_split: int) -> tuple:
        """ Returns the sub clustering for a specific network.

        Parameters
        ----------
        network_i : int
            Indicates which network to return.
        num_networks : int
            The number of networks to define.
        t_split : int
            The number of parcels to split t into.

        Returns
        -------
        v : array of shape (n_targets)
        all_cluster_inds : list of arrays
        """
        assert (network_i > 0)
        network_splits = self.get_clustering_split(num_networks, t_split, network_i)
        rng = range(network_splits[0], network_splits[1])

        clustering = self.get_clustering(num_clusters =t_split, match_colors =False)

        n_targets = self.get_linkage().shape[0] + 1
        v = np.zeros(n_targets)
        all_cluster_inds = []
        for i, cluster in enumerate(rng):
            cluster_inds = np.array(clustering==cluster)
            v[clustering==cluster] = i+1
            all_cluster_inds.append(cluster_inds)
        all_cluster_inds = np.vstack(all_cluster_inds)

        return v, all_cluster_inds


    def get_clustering_match(self, num_clusters: int, ref_linkage, ref_cluster_colors=None,
                             weights=None, ref_weights=None) -> np.ndarray:
        """
        Parameters
        ----------
        num_clusters : int
            The number of clusters in the clustering solution.
        ref_linkage : Linkage
            The reference linkage to match to
        ref_cluster_colors : list of int
            The cluster colors for the reference linkage.
        weights : array of shape (n_features, n_targets)
            If given, match colors according to similarity in weights.

        Returns
        -------
        clustering : array of shape (n_targets)
            The cluster assignment for each target.
        """
        # get clustering
        clustering = self.get_clustering(num_clusters, match_colors =False)

        # get reference linkage
        ref_= ref_linkage # Linkage(ref_linkage, self.brainspace)
        ref_clustering = ref_.get_clustering(num_clusters, match_colors =False)

        new_labels = match_clusters(ref_clustering, clustering, ref_weights, weights)

        return relabel_clusters(clustering, new_labels)


    def get_corrmat(self, num_clusters: int, X: np.ndarray, cluster_order=None, as_avgs: bool = True, metric= Metrics.Correlation) -> np.ndarray:
        """ Returns pairwise correlation matrix.

        Parameters
        ----------
        num_clusters : int
        X : array of shape (n_targets, n_features)
        cluster_order : list of int
        as_avgs : bool

        Returns
        -------
        corrmat : array of shape (n_targets, n_targets)
        """
        if as_avgs:
            if X.shape[0] == 20484:
                X_,_ = self.get_clustering_avgs(num_clusters, X, cluster_order)
            else:
                X_ = X
        else:
            clustering = self.get_clustering(num_clusters, match_colors =False)
            if cluster_order is None:
                cluster_order = np.unique(clustering)

            X_ = []
            for cluster in cluster_order:
                X_.append(X[clustering==cluster,:])
            X_ = np.vstack(X_)

        corrmat = get_dists(X_, as_mat=True, metric=str(metric.value))
        if metric == "euclidean":
            corrmat /= np.max(corrmat)
        corrmat = 1. - corrmat
        return corrmat


    def map_cluster_values_to_voxels(self, cluster_values: np.ndarray, cluster_order=None) -> np.ndarray:
        """
        Maps cluster values to units (voxels or vertices)

        Parameters
        ----------
        cluster_values : array of shape (n_clusters)
        cluster_order : list of int
        """
        num_clusters = len(cluster_values)
        clustering = self.get_clustering(num_clusters, match_colors =False)
        if cluster_order is None:
            cluster_order = np.unique(clustering)

        data = np.zeros(self._num_voxels)
        for c_i, cluster in enumerate(cluster_order):
            data[clustering==cluster] = cluster_values[c_i]
        return data

    def average_data_to_clusters(self, data: np.ndarray, num_clusters: int) -> np.ndarray:
        """ Plots data on a flatmap.

        Parameters
        ----------
        data : array of shape (n_targets)
            The data, one value per vertex.
        num_clusters : int
            Optional, the number of networks to average data within.
        """

        # plot cluster average values
        clustering = self.get_clustering(num_clusters, match_colors =False)
        v = np.zeros(data.shape)
        for t_i in np.unique(clustering):
            v[clustering==t_i] = np.mean(data[clustering==t_i])
        return v