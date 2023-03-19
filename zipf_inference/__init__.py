import numpy as np
from zipf_inference.utils import GroundTruthCounter
import json
import os
import copy
from zipf_inference.globals import DATA_DIR

import torch
import itertools
chain=itertools.chain.from_iterable

from torch.utils.data import Dataset

NL_DESC={
    'p4': (
        'counter from 15 to 301, skip by 15. For every counter, move forward by '
        'counter and turn left 90. Repeat 4 times.'),
    'hoc18': 'if path is forward, move forward. Otherwise, turn left. Repeat.',
    'hoc4': 'Move forward, turn left, move forward, turn right, and move forward.',
    'cip_mask_warmup': '',
    'cip_mask_frame': '',
    'cip_mask_shelter': '',
    'cip_mask_piles': '',
}

def pad_tensor(A, pad_len, fill=0):
    shape = A.shape
    if len(shape) > 1:
        p_shape = copy.deepcopy(list(shape))
        p_shape[0] = pad_len
        P = torch.zeros(*p_shape, dtype=A.dtype) + fill
    else:
        P = torch.zeros(pad_len, dtype=A.dtype) + fill
    A = torch.cat([A, P], dim=0)
    return A

class ExpDataset:

    def __init__(self, dname):
        """
            the following files must be in the dname dir:
                - rank_cnts.json
                - rank_resps.json
                - ranked_resps.json (ordered list of responses)
            to create model_datasets:
                - vocab.json
                - tokenized.json (list of lists)
        """
        dname = os.path.join(DATA_DIR, dname)
        self.dname = dname
        rank_cnts = json.load(open(os.path.join(dname, 'rank_cnts.json')))
        rank_cnts = dict((int(k), v) for k,v in rank_cnts.items())
        rank_resps = json.load(open(os.path.join(dname, 'rank_resps.json')))
        rank_resps = dict((int(k), v) for k,v in rank_resps.items())
        # ranked_resps is a list
        self.ranked_resps = json.load(open(os.path.join(dname, 'ranked_resps.json')))
        total = sum(rank_cnts.values())
        self.sample_p = dict((k,v/total) for k,v in rank_cnts.items()) # used for sampling responses

        self.has_canonical = os.path.isfile(os.path.join(dname, 'raw_to_canonical.json'))
        self.nominal_counts = GroundTruthCounter(rank_cnts) # raw resp counts
        if self.has_canonical:
            # there is a canonical conversion
            self.raw_to_canonical = json.load(os.path.join(dname, 'raw_to_canonical.json'))
            self.canonical_cnts = json.load(os.path.join(dname, 'canonical_cnts.json'))
            self.gt_counts = dict((r,self.canonical_cnts[self.raw_to_canonical[v]])
                                  for r,v in self.rank_cnts.items())
        else:
            self.raw_to_canonical = None
            self.canonical_cnts = None
            self.gt_counts = rank_cnts # used for ranking
        self.rank_resps = rank_resps

        if os.path.isfile(os.path.join(dname, 'vocab.json')):
            self.vocab = json.load(open(os.path.join(dname, 'vocab.json')))
            self.ranked_tokenized = json.load(open(os.path.join(dname, 'tokenized.json')))
        else:
            self.vocab = None
            self.ranked_tokenized = None

    def get_random_samples(self, size):
        """
            "idxs" are sorted by rank (ties broken arbitrarily)
        """

        # idxs start by 1!!
        ranks = np.random.choice(list(self.sample_p.keys()),
                                size=size, p=list(self.sample_p.values()))
        ranks = np.sort(ranks)
        unique_ranks, unique_locs = np.unique(ranks, return_index=True) # np.unique sorts lexicographically
        if self.ranked_tokenized is not None:
            tokenized = [self.ranked_tokenized[i-1] for i in unique_ranks]
        else:
            tokenized = None
        dataset = ModelDataset(self, unique_ranks)
        sample = {
            "idxs": ranks,               # starts with 1, assume sorted
            "unique_idxs": unique_ranks, # starts with 1, assume sorted (by np.unique sorted)
            "unique_locs": unique_locs,  # indexing within idx
            "model_dataset": dataset,
            "tokenized": tokenized
        }
        return sample

    def get_tokenized_input(self, samples):
        pass

class ModelDataset(Dataset):
    """
        Dataset constructed from sampled responses and used for model fitting
        Uses canonical (masked) responses if they exist
    """

    def __init__(self, exp_dataset, unique_idxs=None, max_len=200):
        self.exp_dataset = exp_dataset
        if unique_idxs is None:
            self.original_ranks = list(exp_dataset.rank_resps.keys())
        else:
            self.original_ranks = unique_idxs # this is a list of "ranks"
        self.max_len = max_len

    def __len__(self):
        return len(self.original_ranks)

    def __getitem__(self, i):
        raw_input = self.exp_dataset.ranked_resps[self.original_ranks[i]-1]
        if self.exp_dataset.has_canonical:
            canonical_input = self.exp_dataset.raw_to_canonical[raw_input]
        else:
            canonical_input = None
        token_ids = self.exp_dataset.ranked_tokenized[self.original_ranks[i]-1]
        token_ids = np.asarray(self.exp_dataset.ranked_tokenized[self.original_ranks[i]-1],
                                dtype=np.long)
        original_rank = self.original_ranks[i]

        item = dict(
            original_rank=original_rank,
            original_freq=self.exp_dataset.gt_counts[original_rank],
            raw_inp=raw_input,
            canonical_inp=canonical_input,
            length = len(token_ids),
            token_ids = torch.from_numpy(token_ids).long()
        )
        return dict(src=item, tgt=item)

    @classmethod
    def collate_fn(self, batch):
        items = dict()
        for source in ['src', 'tgt']:
            token_ids_lst = [b[source]['token_ids'] for b in batch]
            max_len = max(len(token_ids) for token_ids in token_ids_lst)
            token_ids_lst = [pad_tensor(token_ids, max_len-len(token_ids)) for token_ids in token_ids_lst]
            token_ids = torch.stack(token_ids_lst)

            items[source] = dict(
                original_rank = [b[source]['original_rank'] for b in batch],
                original_freq = [b[source]['original_freq'] for b in batch],
                raw_inp = [b[source]['raw_inp'] for b in batch],
                length = [b[source]['length'] for b in batch],
                token_ids = token_ids
            )
        return items

    @classmethod
    def stitch_batches(self, batches):
        keys = ['original_rank', 'original_freq', 'raw_inp', 'length', 'token_ids']
        items = dict()
        for source in ['src', 'tgt']:
            items[source] = {}
            for key in keys:
                items[source][key] = chain([b[source][key] for b in batches])
        return items

if __name__=='__main__':
    ExpDataset('hoc18')
