import numpy as np
import pickle
import os
from collections import Counter
from tqdm.autonotebook import trange

from zipf_inference.utils import clear_memory, edit_distance
from zipf_inference.globals import SCORER_DIR

import nltk
import torch
import matplotlib.pyplot as plt

# TODO: should the scorer know model?

class SimilarityScorer:
    """
        computes ranking score based on a vector-based measure of similarity.
    """

    def __init__(self, model, debug=False):
        self.device = model.device
        self.model = model
        self.debug_info = {}
        self.debug= debug
        self.curr_call_label = None
        self.full_sim_mat = None

    def __call__(self, func_type, sampled_resps, call_label='', **kwargs):
        score_method = getattr(self, f'score_{func_type}')
        self.curr_call_label = call_label
        self.debug_info[call_label] = {}
        score_out = score_method(sampled_resps=sampled_resps, **kwargs)
        self.curr_call_label = None
        return score_out

    def compute_and_save_sim_mat(self, sim_fname):
        self.full_sim_mat = self.compute_similarity(self.model.embs)
        sim_fname = os.path.join(SCORER_DIR, 'bin', sim_fname) + '.sim'
        pickle.dump(self.full_sim_mat, open(sim_fname, 'wb'))

    def compute_similarity(self, embs):
        raise NotImplementedError

    def get_sub_similarity(self, sampled_resps):
        return self.compute_similarity(self.model.embs)

    def score_submissions_nbsize(self, threshold, sampled_resps=None, embs=None):
        """
            embs and idxs are sorted against each other
        """
        sub_similarity = self.get_sub_similarity(sampled_resps)
        sub_similarity = (sub_similarity - sub_similarity.min())/(sub_similarity.max() - sub_similarity.min())
        nb_size = (sub_similarity > threshold).sum(axis=0)
        sorted_idxs = np.flip(np.argsort(nb_size))

        ordered_idxs = sampled_resps['unique_idxs'][sorted_idxs]
        ordered_score = nb_size[sorted_idxs]
        return ordered_idxs, ordered_score

    def score_freq_topk_sum(self, sampled_resps=None, multiplicity=False, bottomk=False, k=10, absolute=False):
        similarity = self.get_sub_similarity(sampled_resps)
        if absolute:
            similarity = np.abs(similarity)

        if multiplicity:
            unique_idxs = sampled_resps['unique_idxs']
            idxs = sampled_resps['idxs']
            dup_locs = np.searchsorted(unique_idxs, idxs)
            similarity = similarity[np.ix_(dup_locs, dup_locs)]

        if k is None:
            k = similarity.shape[0]
        idxs = sampled_resps['idxs']
        unique_idxs = sampled_resps['unique_idxs']
        counts = Counter(idxs)
        scores = np.array([counts[code] for code in unique_idxs])
        head_loc = (scores > 1)
        head_scores, _ = scores[head_loc], scores[~head_loc]
        head_idxs, body_idxs = unique_idxs[head_loc], unique_idxs[~head_loc]
        head_sorting_idx = np.argsort(head_scores)[::-1]
        head_ordered_idxs = head_idxs[head_sorting_idx]
        head_ordered_scores = head_scores[head_sorting_idx].astype(float)

        body_loc_idx = (~head_loc).nonzero()[0]


        if bottomk:
            top_k_sum = np.sort(similarity, axis=1)[:,:k].sum(axis=1)
            top_k_idx = np.flip(np.argsort(similarity, axis=1), axis=1)[:,-k:]
        else:
            top_k_sum = np.sort(similarity, axis=1)[:,-k:].sum(axis=1)
            top_k_idx = np.flip(np.argsort(similarity, axis=1), axis=1)[:,:k]

        if multiplicity:
            unique_loc = sampled_resps['unique_locs']
            top_k_sum = top_k_sum[unique_loc]
            top_k_idx = top_k_idx[unique_loc]


        if self.debug:
            call_label = self.curr_call_label
            self.debug_info[call_label]['topk_idx'] = top_k_idx

        body_sorting_idx = np.flip(np.argsort(top_k_sum[body_loc_idx]))

        body_ordered_idxs = body_idxs[body_sorting_idx]
        body_ordered_scores = top_k_sum[body_loc_idx][body_sorting_idx]

        if len(head_ordered_scores) > 0:
            head_ordered_scores += float(max(body_ordered_scores)-min(head_ordered_scores)+1.0)
        ordered_idxs = np.concatenate([head_ordered_idxs, body_ordered_idxs])
        ordered_scores = np.concatenate([head_ordered_scores, body_ordered_scores])
        return ordered_idxs, ordered_scores

    def score_submissions_topk_sum(self, sampled_resps=None, bottomk=False, k=10, absolute=False):
        raise Exception('abs + multiplicity fix')
        sub_similarity = self.get_sub_similarity(sampled_resps)

        if absolute:
            sub_similarity = np.absolute(sub_similarity)

        if bottomk:
            top_k_sum = np.sort(sub_similarity, axis=1)[:,:k].sum(axis=1)
            top_k_idx = np.argsort(sub_similarity, axis=1)[:,:k]
        else:
            top_k_sum = np.sort(sub_similarity, axis=1)[:,-k:].sum(axis=1)
            top_k_idx = np.flip(np.argsort(sub_similarity, axis=1)[:,-k:], axis=1)

        if self.debug:
            call_label = self.curr_call_label
            self.debug_info[call_label]['topk_idx'] = top_k_idx
        sorted_idxs = np.flip(np.argsort(top_k_sum))

        ordered_idxs = sampled_resps['unique_idxs'][sorted_idxs]
        ordered_score = top_k_sum[sorted_idxs]
        return ordered_idxs, ordered_score

    def score_dist_to_sample_head(self, head, sampled_resps=None, absolute=False):
        sub_similarity = self.get_sub_similarity(sampled_resps)

        if absolute:
            sub_similarity = np.absolute(sub_similarity)
        head_sum = sub_similarity.sum(axis=1)
        sorted_idxs = np.flip(np.argsort(head_sum))

        ordered_idxs = sampled_resps['unique_idxs'][sorted_idxs]
        ordered_score = head_sum[sorted_idxs]
        return ordered_idxs, ordered_score

class PreComputedSimilarityScorer(SimilarityScorer):

    def __init__(self, sim_fname, **kwargs):
        super().__init__(**kwargs)
        sim_fname = os.path.join(SCORER_DIR, 'bin', sim_fname) + '.sim'
        self.full_sim_mat = pickle.load(open(sim_fname, 'rb'))

    def get_sub_similarity(self, sampled_resps):
        unique_idxs = sampled_resps['unique_idxs']
        return self.full_sim_mat[np.ix_(unique_idxs-1, unique_idxs-1)]

class L2DistScorer(SimilarityScorer):
    LABEL='l2'

    def compute_similarity(self, embs):
        embs = embs.to(self.device)
        # embs = torch.from_numpy(embs).to(self.device)
        total_len = embs.size(0)
        batch_size = 200
        steps = int(np.ceil(total_len/batch_size))
        l2_dist = []
        for i in trange(steps, disable=(steps < 5)):
            y = embs[batch_size*i:batch_size*(i+1)]
            sim = torch.cdist(embs, y)
            l2_dist.append(sim.detach().cpu().numpy())
        del embs
        clear_memory()
        l2_sim = -np.concatenate(l2_dist, axis=1)
        return l2_sim

class CosineSimScorer(SimilarityScorer):
    LABEL='cos'

    def compute_similarity(self, embs):
        embs = embs.to(self.device)
        # embs = torch.from_numpy(embs).to(self.device)
        total_len = embs.size(0)
        batch_size = 200
        steps = int(np.ceil(total_len/batch_size))
        cos_sim = []
        for i in trange(steps, disable=(steps < 5)):
            y = embs[batch_size*i:batch_size*(i+1)].T
            sim = torch.mm(embs, y)
            x_l2 = embs.unsqueeze(dim=1).norm(dim=2)
            y_l2 = y.T.unsqueeze(dim=0).norm(dim=2)
            sim = sim / (x_l2 * y_l2)
            sim = sim.detach().cpu().numpy()
            cos_sim.append(sim)
        del embs
        clear_memory()
        dot_sim = np.concatenate(cos_sim, axis=1)
        # dot_sim = torch.cat(dot_sim, dim=1).detach().cpu().numpy()
        clear_memory()
        return dot_sim


class DotProductScorer(SimilarityScorer):
    LABEL='dot'

    def compute_similarity(self, embs):
        embs = embs.to(self.device)
        # embs = torch.from_numpy(embs).to(self.device)
        total_len = embs.size(0)
        batch_size = 200
        steps = int(np.ceil(total_len/batch_size))
        dot_sim = []
        for i in trange(steps, disable=(steps < 5)):
            y = embs[batch_size*i:batch_size*(i+1)].T
            sim = torch.mm(embs, y).detach().cpu().numpy()
            dot_sim.append(sim)
        del embs
        clear_memory()
        dot_sim = np.concatenate(dot_sim, axis=1)
        # dot_sim = torch.cat(dot_sim, dim=1).detach().cpu().numpy()
        clear_memory()
        return dot_sim

class EditDistanceScorer(SimilarityScorer):
    LABEL='ed'

    def compute_similarity(self, embs):
        # embs are assumed to be list of lists (of tokens)
        if 'edit_distance' not in self.model._cache:
            self.model._cache['edit_distance'] = edit_distance(embs)
        ed_score = -self.model._cache['edit_distance']
        if self.debug:
            plt.imshow(ed_score)
            plt.show()
        # raise Exception('resolve the smaller-the-closer score issue')
        return ed_score
