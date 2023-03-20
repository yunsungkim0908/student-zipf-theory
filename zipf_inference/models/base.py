import torch.nn as nn
import pickle
import os

from zipf_inference.globals import MODEL_DIR
from zipf_inference.utils import check_nan, normalize_embs

# TODO: clean memory after every fit

class BaseModel(nn.Module):

    def __init__(self, device, exp=None, exp_dataset=None):
        super().__init__()
        self.device = device
        self.exp_dataset = exp_dataset if exp is None else exp.dataset
        self.scores = {}
        self.scorers = {}
        self.scorers_call_kwargs = {}
        self.embs = None

    def initialize_model(self):
        """
            shall be called for every new sample.
            cache stores values to be re-used.
        """
        self._cache = {}

    def get_and_reset_scores(self):
        return_scores = self.scores
        self.scores = {}
        return return_scores

    def set_many_scorers(self, config_dict):
        for name, config in config_dict.items():
            self.set_scorer(name, **config)

    def set_scorer(self, name=None,
                   scorer_class=None, scorer_func=None, func_type='',
                   call_kwargs=None, debug=False, init_kwargs={}):
        if name in self.scorers:
            if name is None:
                raise Exception('cannot reset unnamed scorer')
            raise Exception(f'{name} scorer already set')

        assert (scorer_class is None) != (scorer_func is None)

        call_kwargs = {} if call_kwargs is None else call_kwargs
        if debug:
            init_kwargs['debug'] = debug

        if scorer_class is not None:
            call_kwargs['func_type'] = func_type
            self.scorers[name] = scorer_class(model=self, **init_kwargs)
        else:
            self.scorers[name] = scorer_func
        self.scorers_call_kwargs[name] = call_kwargs

    def call_scorer(self, scorer_name, sampled_resps, call_label=''):
        """ Invokes scorer for the current value of embs """
        call_name = scorer_name + ('' if len(call_label) == 0 else f'_{call_label}')
        if call_name in self.scores:
            raise Exception('calling scorer twice')
        kwargs = self.scorers_call_kwargs[scorer_name]
        if hasattr(self.scorers[scorer_name], 'model'):
            kwargs['call_label'] = call_label
        scorer = self.scorers[scorer_name]
        self.scores[call_name] = scorer(sampled_resps=sampled_resps,
                                        **kwargs)

    def fit(self, sampled_resps):
        """
            process the sampled responses and invoke scorers
            it is assumed that this function sets the `self.embs` attribute
            which the scorer uses to compute scores
        """
        raise NotImplementedError

class TokenBasedModel(BaseModel):

    def fit(self, sampled_resps):
        self.embs = sampled_resps['tokenized']
        for scorer_name in self.scorers.keys():
            self.call_scorer(scorer_name, sampled_resps)

class NullModel(BaseModel):

    def fit(self, sampled_resps):
        for scorer_name in self.scorers.keys():
            self.call_scorer(scorer_name, sampled_resps)

class PreComputedEmbsModel(NullModel):

    def __init__(self, embs_fname, norm_emb=False, **kwargs):
        super().__init__(**kwargs)
        self.norm_emb = norm_emb
        self.full_embs = pickle.load(open(os.path.join(MODEL_DIR, 'bin', f'{embs_fname}.emb'), 'rb'))

    def fit(self, sampled_resps):
        self.embs = self.full_embs[sampled_resps['unique_idxs']-1]
        if self.norm_emb:
            self.embs = normalize_embs(self.embs)
        for scorer_name in self.scorers.keys():
            self.call_scorer(scorer_name, sampled_resps)


class PretrainedModel(BaseModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.setup_pretrained()
        all_inputs = self.convert_all_inputs_to_tensors()
        self.embs = self.compute_all_embeddings(all_inputs)

    def setup_pretrained(self):
        raise NotImplementedError

    def fit(self, sampled_resps, **kwargs):
        pass

    def save_embeddings(self, emb_fname):
        emb_fname = os.path.join(MODEL_DIR, 'bin', emb_fname) + '.emb'
        pickle.dump(self.embs, open(emb_fname, 'wb'))
