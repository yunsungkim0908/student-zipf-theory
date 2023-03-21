import gc
import torch
import numpy as np
from scipy.special import loggamma
import os
import nltk
from collections import Counter

def check_nan(tensor, name):
    if tensor.isnan().any():
        print(name, tensor)

def edit_distance(embs):
    assert isinstance(embs, list)

    edit_distance = []
    for i, tok_i in enumerate(embs):
        row = []
        for j, tok_j in enumerate(embs):
            if i == j:
                row.append(0)
            elif i > j:
                row.append(edit_distance[j][i])
            else:
                row.append(nltk.edit_distance(tok_i, tok_j))
        edit_distance.append(row)
    edit_distance = np.asarray(edit_distance)
    return edit_distance

def normalize_embs(embs):
    mu, std = embs.mean(dim=0, keepdim=True), embs.std(dim=0, keepdim=True)
    std = std.clamp(min=1e-5)
    embs = (embs - mu) / std
    return embs

class GroundTruthCounter:
    def __init__(self, cnts):
        self.cnts = cnts
        self.ordered_codes = np.array(sorted(cnts.keys(), key=lambda x: cnts[x], reverse=True))
        self.ordered_counts = np.array(sorted(cnts.values(), reverse=True))
        self.set_tail()

    def set_tail(self, frac=0.1):
        """
            this is maximum tail count limit which makes the tail fraction >= 0.2
        """
        total = sum(self.cnts.values())
        count_of_counts = Counter(self.cnts.values())
        counts = np.sort(np.unique(list(count_of_counts.keys())))
        count_of_counts = [count_of_counts[x] for x in counts]
        not_tail = (np.cumsum(counts*count_of_counts) >= frac*total)
        tail_limit_cnt = counts[np.nonzero(not_tail)[0].min()]
        self.tail_limit = tail_limit_cnt

    def __getitem__(self, idx):
        return self.cnts[idx]

    def __len__(self):
        return len(self.cnts)

class GroundTruthProb:

    def __init__(self, dirname, gram_type='vam'):
        self.dirname = dirname
        # rank_freq = np.fromfile(os.path.join(dirname, 'sorted_vals.bin'), dtype=np.int32)
        # self.rank_p = rank_freq / rank_freq.sum()
        self.size = np.fromfile(os.path.join(dirname, 'sizes.bin'), dtype=np.int32)
        if len(self.size) == 4:
            self.size = self.size[:-1]
        assert len(self.size) == 3
        self.dist = np.fromfile(os.path.join(dirname, 'dists.bin'), dtype=np.double).reshape(self.size)
        self.gram_type = gram_type

    def _get_log_prob(self, seq):
        if self.gram_type == 'vam':
            return self._get_unif_ablity_cont_stateful_log_prob(seq)

    def _get_unif_ablity_cont_stateful_log_prob(self, seq):
        seq = seq_to_tuple(seq)
        assert len(seq) == self.size[0]
        log_p = 0.0
        zero_count = 0
        for i in range(len(seq)):
            if seq[i] != 0:
                log_p += np.log(self.dist[i][0][seq[i]-1])
            else:
                zero_count += 1
        log_p += loggamma(zero_count+1) + loggamma(len(seq)-zero_count+1) - loggamma(len(seq)+2)
        return log_p


    def __getitem__(self, x):
        return self._get_log_prob(x)

def get_slope(dirname):
    sorted_vals_fname = os.path.join(dirname, 'sorted_vals.bin')
    sorted_vals = np.fromfile(sorted_vals_fname, dtype=np.int32)
    max_x = min(1e7, len(sorted_vals)-1)
    slope = (np.log(sorted_vals[int(1e3)] - np.log(sorted_vals[int(max_x)])))/(np.log(1e3)-np.log(max_x))
    return slope

def seq_to_tuple(seq):
    return tuple([int(c) for c in seq.split('|') if len(c.strip()) > 0])

def clear_memory():
    gc.collect()
    torch.cuda.empty_cache()

def argsort_into(a, b):
    a_to_sorted = np.argsort(a)
    sorted_to_b = np.searchsorted(a[a_to_sorted], b)
    return a_to_sorted[sorted_to_b]

def get_gt_head_body_tail(gt_counts, unique_codes):
    ordered_codes = sorted(unique_codes, key=lambda x: gt_counts[x], reverse=True)
    head = ordered_codes[:10]
    tail = np.array([code for code in unique_codes if gt_counts[code] <= gt_counts.tail_limit])
    body = np.array([code for code in unique_codes if code not in head and code not in tail])
    return head, body, tail
