import numpy as np
from random import shuffle
from collections import Counter
from zipf_inference.utils import seq_to_tuple

def get_hybrid_scores(sampled_resps, ordered_idxs, ordered_scores):
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

    body_loc_idxs = np.isin(ordered_idxs, body_idxs)

    body_ordered_idxs = ordered_idxs[body_loc_idxs]
    body_ordered_scores = ordered_scores[body_loc_idxs]

    if len(head_ordered_scores) > 0:
        head_ordered_scores += float(max(body_ordered_scores)-min(head_ordered_scores) + 1.0)

    ordered_idxs = np.concatenate([head_ordered_idxs, body_ordered_idxs])
    ordered_scores = np.concatenate([head_ordered_scores, body_ordered_scores])
    return ordered_idxs, ordered_scores

def compute_ranks(ordered_score, average_only=False):
    """
        larger score means higher rank
        ranks start with 0
    """
    unique_scores, score_counts = np.unique(ordered_score, return_counts=True)
    score_sorting_idx = np.argsort(unique_scores)[::-1]
    ordered_score_counts = score_counts[score_sorting_idx]
    left_ranks = []
    right_ranks = []
    tie_break_ranks = []
    for i in range(len(ordered_score_counts)):
        count = ordered_score_counts[i]
        left_rank = len(left_ranks)
        right_rank = left_rank + count
        for _ in range(count):
            left_ranks.append(left_rank)
            right_ranks.append(right_rank)
        ties = list(range(left_rank, right_rank))
        shuffle(ties)
        tie_break_ranks.extend(ties)
    left_ranks = np.array(left_ranks)
    right_ranks = np.array(right_ranks)
    avg_ranks = (left_ranks + right_ranks - 1) / 2
    if average_only:
        return avg_ranks
    return dict(
        left_ranks=left_ranks,
        right_ranks=right_ranks,
        avg_ranks=avg_ranks,
        tie_break_ranks=tie_break_ranks
    )

def score_frequency(sampled_resps, ground_truth_cnts=None):
    idxs = sampled_resps['idxs']
    unique_idxs = sampled_resps['unique_idxs']
    # use ground truth counts as scorer if provided
    counts = Counter(idxs) if ground_truth_cnts is None else ground_truth_cnts
    scores = np.array([counts[code] for code in unique_idxs])
    sorting_idx = np.argsort(scores)[::-1]
    ordered_idxs = sampled_resps['unique_idxs'][sorting_idx]
    ordered_scores = scores[sorting_idx]
    return ordered_idxs, ordered_scores

def score_freq_head_rand_tail(sampled_resps):
        idxs = sampled_resps['idxs']
        unique_idxs = sampled_resps['unique_idxs']
        counts = Counter(idxs)
        scores = np.array([counts[code] for code in unique_idxs])
        head_loc = (scores > 1)
        head_idxs, tail_idxs = unique_idxs[head_loc], unique_idxs[~head_loc]
        head_scores = scores[head_loc] + len(tail_idxs)

        head_sorting_idx = np.argsort(head_scores)[::-1]
        head_ordered_idxs = head_idxs[head_sorting_idx]
        head_ordered_scores = head_scores[head_sorting_idx].astype(float)

        tail_sorting_idx = np.random.permutation(len(tail_idxs))
        tail_ordered_idxs = tail_idxs[tail_sorting_idx]
        tail_ordered_scores = np.arange(len(tail_idxs)-1,-1,-1)

        ordered_idxs = np.concatenate([head_ordered_idxs, tail_ordered_idxs])
        ordered_scores = np.concatenate([head_ordered_scores, tail_ordered_scores])
        return ordered_idxs, ordered_scores

def score_rand_perm(sampled_resps):
    unique_idxs = sampled_resps['unique_idxs']
    rand_perm = np.random.permutation(len(unique_idxs))
    ordered_idxs, ordered_scores = unique_idxs[rand_perm], np.arange(len(unique_idxs), 0, -1) - 1
    return ordered_idxs, ordered_scores

def score_frac_singleton(sampled_resps):
    idxs = sampled_resps['idxs']
    unique_idxs = sampled_resps['unique_idxs']
    return np.arange(len(unique_idxs)), len(unique_idxs)/len(idxs)

def code_score_dist_topk_sum(sampled_resps, k=10, metric='l1'):
    raise NotImplementedError

def code_score_freq_dist_topk_sum(sampled_resps, k=10, metric='l1'):
    raise NotImplementedError

