import numpy as np
from zipf_inference.scorer.base_scorer import get_hybrid_scores

class TokenBasedScorer:

    def __init__(self, model, debug=False):
        self.device = model.device

    def __call__(self, func_type, sampled_resps, call_label='', **kwargs):
        score_method = getattr(self, f'score_{func_type}')
        ordered_idxs, ordered_score = score_method(sampled_resps=sampled_resps, **kwargs)
        score_out = get_hybrid_scores(sampled_resps, ordered_idxs, ordered_score)
        return score_out


    def score_length(self, sampled_resps):
        tokens = sampled_resps['tokenized']
        lengths = np.array([len(toks) for toks in tokens])
        sorted_idxs = np.argsort(lengths)

        ordered_idxs = sampled_resps['unique_idxs'][sorted_idxs]
        ordered_score = lengths[sorted_idxs]
        return ordered_idxs, ordered_score

    def score_length_diff_from_mean(self, sampled_resps):
        tokens = sampled_resps['tokenized']
        lengths = np.array([len(toks) for toks in tokens])
        mean_length = np.mean(lengths)
        diff_to_mean = [abs(length - mean_length) for length in lengths]
        sorted_idxs = np.argsort(diff_to_mean)

        ordered_idxs = sampled_resps['unique_idxs'][sorted_idxs]
        ordered_score = lengths[sorted_idxs]
        return ordered_idxs, ordered_score
