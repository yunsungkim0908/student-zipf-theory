from mlpack import det
from sklearn.neighbors import KernelDensity
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from zipf_inference.utils import normalize_embs, edit_distance
from zipf_inference.scorer.base_scorer import get_hybrid_scores
from collections import Counter
import nltk
import torch
import numpy as np

EPS=1e-5
KERNEL={
    'gaussian': lambda x: np.exp(-x**2/2),
    'tophat': lambda x: (x < 1).astype('float'),
    'epanechnikov': lambda x: np.maximum(1 - x**2, 0),
    'linear': lambda x: np.maximum(1 - x, 0),
    'cosine': lambda x: np.maximum(np.cos((np.pi/2)*x), 0)
}

class DensityEstimatorScorer:

    def __init__(self, model, debug=False):
        self.model = model
        self.debug = debug

    def compute_distance(self, embs, unit_norm=True):
        if isinstance(embs, np.ndarray):
            n,_ = embs.shape
            embs = embs / np.linalg.norm(embs, axis=1).max()
            if unit_norm:
                embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)
            dist = np.linalg.norm(
                np.expand_dims(embs, 1) - np.expand_dims(embs, 0), axis=-1)
        elif isinstance(embs, list):
            # if embs is a list, then it's a list of tokens.
            # in this case we assume edit distance
            if 'edit_distance' not in self.model._cache:
                self.model._cache['edit_distance'] = edit_distance(embs)
            dist = self.model._cache['edit_distance']

        return dist

    def __call__(self, sampled_resps, call_label='', multiplicity=True, hybrid=True, **kwargs):
        # no label saving yet
        embs = self.model.embs
        if isinstance(embs, torch.Tensor):
            embs = self.model.embs.detach().cpu().numpy()

        if multiplicity:
            unique_idxs = sampled_resps['unique_idxs']
            idxs = sampled_resps['idxs']
            dup_locs = np.searchsorted(unique_idxs, idxs)
            _, dedup_locs = np.unique(dup_locs, return_index=True)
            embs = [embs[i] for i in dup_locs]\
                if isinstance(embs, list) else embs[dup_locs]

        density = self.compute_density(embs, **kwargs)
        if multiplicity:
            unique_locs = sampled_resps['unique_locs']
            density = density[unique_locs]

        sorted_idxs = np.flip(np.argsort(density))
        ordered_idxs = sampled_resps['unique_idxs'][sorted_idxs]
        ordered_score = density[sorted_idxs]
        if hybrid:
            ordered_idxs, ordered_score = get_hybrid_scores(
                sampled_resps, ordered_idxs, ordered_score)
        return ordered_idxs, ordered_score

class DETreeScorer(DensityEstimatorScorer):

    def compute_density(self, embs, **kwargs):
        pca = PCA(n_components=2)
        embs = pca.fit_transform(embs)
        embs = normalize_embs(torch.tensor(embs)).numpy()
        de_tree = det(
            folds=10, input_model=None, max_leaf_size=1,
            min_leaf_size=1, path_format='lr', skip_pruning=True,
            test=embs, training=embs, verbose=False
        )
        self.de_tree = de_tree

        density = de_tree['training_set_estimates'].squeeze()

        return density

class GeneralizedKNNScorer(DensityEstimatorScorer):

    def compute_density(self, embs, func_type, k=10, unit_norm=True):
        dist = self.compute_distance(embs, unit_norm=unit_norm)

        src_window_width = np.sort(dist, axis=1)[:,k+1:k+2] # k-th neighbor dist from src
        src_window_width = np.clip(src_window_width, a_min=EPS, a_max=None)
        kernel_inp = dist/src_window_width
        kernel_comps = KERNEL[func_type](kernel_inp)/src_window_width
        density = kernel_comps.mean(axis=1)

        return density

class VariableKDEScorer(DensityEstimatorScorer):

    def compute_density(self, embs, func_type, k=10, unit_norm=True, bandwidth=1):
        h = bandwidth
        dist = self.compute_distance(embs, unit_norm=unit_norm)

        tgt_window_width = np.sort(dist, axis=0)[k+1:k+2,:] # k-th neighbor dist from tgt
        tgt_window_width = np.clip(tgt_window_width, a_min=EPS, a_max=None)
        kernel_inp = dist/(h*tgt_window_width)
        kernel_comps = KERNEL[func_type](kernel_inp)/(h*tgt_window_width)
        density = kernel_comps.mean(axis=1)

        return density

class FixedBandwidthKDEScorer(DensityEstimatorScorer):

    def compute_density(self, embs, func_type, k=10, unit_norm=True, bandwidth=1):
        h = bandwidth
        dist = self.compute_distance(embs, unit_norm=unit_norm)

        kernel_inp = dist/h
        kernel_comps = KERNEL[func_type](kernel_inp)/h
        density = kernel_comps.mean(axis=1)

        return density

class KDEScorer(DensityEstimatorScorer):

    def compute_density(self, embs, func_type, unit_norm=True, bandwidth=None):
        embs = embs / np.linalg.norm(embs, axis=1).max()
        if unit_norm:
            embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)
        dist = np.linalg.norm(np.expand_dims(embs, 1) - np.expand_dims(embs, 0), axis=-1)
        dist = dist[~np.eye(dist.shape[0], dtype=bool)].reshape(dist.shape[0], -1)
        if bandwidth is None:
            kde = KernelDensity()
            max_min_dist = dist.min(axis=1).max() + 1e-7
            bandwidth_range = np.linspace(max_min_dist, dist.max(), 5)
            # bandwidth_range = np.linspace(
            #     max_min_dist, (max_min_dist + dist.max())/2, 30)
            # print(bandwidth_range.min())
            # print(bandwidth_range.max())
            parameters = {
                'kernel': [func_type],
                'bandwidth': bandwidth_range}
            clf = GridSearchCV(kde, parameters, cv=embs.shape[0])
            kde = clf.fit(embs)
            if self.debug:
                from pprint import pprint
                pprint(clf.best_params_)
            kde = KernelDensity(**clf.best_params_)
        else:
            kde = KernelDensity(kernel=func_type, bandwidth=bandwidth)

        kde.fit(embs)
        return kde.score_samples(embs)
