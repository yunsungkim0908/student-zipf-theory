from collections import defaultdict
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
from itertools import chain
from collections import Counter

from zipf_inference.dataset import ExpDataset
from zipf_inference.globals import OUT_DIR
from zipf_inference.scorer import compute_ranks
from zipf_inference.utils import argsort_into
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from scipy.stats import wasserstein_distance, kendalltau, spearmanr

from tqdm.autonotebook import trange
import traceback

def to_dict(nested_dict):
    if isinstance(nested_dict, PerformanceMeter):
        nested_dict = nested_dict.metric
    if not isinstance(nested_dict, dict):
        return nested_dict
    current = {}
    for key, val in nested_dict.items():
        current[key] = to_dict(val)
    return current

class PerformanceMeter:

    def __init__(self, metric_dict=None):
        self.metric = defaultdict(lambda: defaultdict(list)) if metric_dict is None else metric_dict

    def add_perf(self, config_name, metrics):
        for metric, val in metrics.items():
            self.metric[config_name][metric].append(val)

    def get_average(self, print_vals=None):
        avg = defaultdict(dict)
        for config_name, perf_dict in self.metric.items():
            for metric, val in perf_dict.items():
                avg[config_name][metric] = (np.array(val).mean(), np.array(val).std())

        if print_vals is not None:
            for config_name, perf_dict in avg.items():
                print(f'\t{config_name}')
                for k,(mu, sigma) in perf_dict.items():
                    print(f'\t\t{k}: {mu:.2f} +/- {sigma:.2f}')
        return avg

class Evaluator:
    """
        evaluates the quality of the ranks induced by the scores
    """

    def __init__(self, exp_dataset, sampled_resps, random_tie_break=True, subset_codes=None):
        self.gt_counts = exp_dataset.gt_counts
        self.random_tie_break = random_tie_break
        self.sampled_resps = sampled_resps

        self.reset_ground_truth(subset_codes)

    def reset_ground_truth(self, subset_codes=None):
        """
            finds the ground-truth ordering within the specified subset of codes
            assumes subset_codes is topologically sorted
        """
        self.subset_codes = subset_codes
        unique_codes = subset_codes if subset_codes is not None else self.sampled_resps['unique_idxs']
        sample_code_counter = Counter(self.sampled_resps['idxs'])
        sample_code_counts = np.array([sample_code_counter[code] for code in unique_codes])
        gt_code_counts = np.array([self.gt_counts[code] for code in unique_codes])

        gt_sort_idx = np.argsort(gt_code_counts)[::-1]

        self.raw_gt_ord_codes = unique_codes[gt_sort_idx]
        self.gt_und_cnts = gt_code_counts[gt_sort_idx]
        self.singleton_locs = np.where(sample_code_counts == 1)[0]
        self.gt_und_wgts = {
            'ranks': len(self.raw_gt_ord_codes) - compute_ranks(self.gt_und_cnts)['avg_ranks'],
            'probs': self.gt_und_cnts/self.gt_und_cnts.sum(),
        }

    def induce_dist_and_fix_order(self, raw_pred_ordered_codes, raw_pred_ordered_score):
        """
            rearranges predicted ordering in order to break ties (keeps ground truth intact)
            in favor of better alignment between prediction and ground-truth
        """
        if self.subset_codes is not None:
            subset_idx = np.isin(raw_pred_ordered_codes, self.subset_codes)
            raw_pred_ordered_codes = raw_pred_ordered_codes[subset_idx]
            raw_pred_ordered_score = raw_pred_ordered_score[subset_idx]
        pred_ranks = compute_ranks(raw_pred_ordered_score)
        left_ranks = pred_ranks['left_ranks']

        # break ties in predicted ranks to allow for smaller discrepancy
        pred_ord_codes = []
        curr_group = []
        # TODO: split probabilities for ties?
        for i in range(len(left_ranks)):
            code = raw_pred_ordered_codes[i]
            l_rank = left_ranks[i]
            curr_group.append((code, l_rank))
            if i == len(left_ranks) - 1 or l_rank != left_ranks[i+1]:
                curr_group = sorted([x[0] for x in curr_group],
                                    key=lambda x: self.gt_counts[x], reverse=True)
                if self.random_tie_break:
                    curr_group = list(np.random.permutation(curr_group))
                pred_ord_codes.append(curr_group)
                curr_group = []
        self.pred_ord_codes = np.array(list(chain.from_iterable(pred_ord_codes)))

        # rearrange the predicted scores to match the gt order
        pred_to_gt_idx = argsort_into(self.pred_ord_codes, self.raw_gt_ord_codes)

        assert len(pred_to_gt_idx) == len(np.unique(pred_to_gt_idx))
        self.pred_ind_wgts = {
            'ranks': len(self.raw_gt_ord_codes) - pred_ranks['avg_ranks'][pred_to_gt_idx],
            'probs': self.gt_und_wgts['probs'][pred_to_gt_idx]
        }

    def get_ranking_metric(self, weight='probs'):
        ranks = np.arange(len(self.raw_gt_ord_codes)) + 1
        # rank_disc = np.log(ranks+1)

        metric = {
            'l1': np.absolute(self.gt_und_wgts[weight] - self.pred_ind_wgts[weight]).sum(),
            'emd': wasserstein_distance(ranks, ranks, self.gt_und_wgts[weight], self.pred_ind_wgts[weight])}

        for sub_idxs, header in [(self.singleton_locs, 'unq_'), (None, '')]:
            if sub_idxs is None:
                sub_idxs = np.arange(len(self.raw_gt_ord_codes))
            gt_probs = self.gt_und_wgts['probs'][sub_idxs]
            pred_probs = self.pred_ind_wgts['probs'][sub_idxs]

            worst_log_l1 = np.absolute(np.log(gt_probs) - np.log(np.flip(gt_probs))).sum()

            metric.update({
                f'{header}norm_log_l1': np.absolute(np.log(gt_probs) - np.log(pred_probs)).sum() / worst_log_l1,
            })

        return metric

    def plot_predictions(self, label, weight='probs', ylogscale=False, show_plot=True, savefig=False):
        dist = self.get_distance(weight)
        xvals = np.arange(len(self.raw_gt_ord_codes))
        plt.figure(figsize=(12,6))
        plt.title(f'Ground-Truth vs Inferred Ranks ({len(self.sampled_resps["idxs"])} Samples)\n{weight.title()}')
        plt.bar(xvals, self.gt_und_wgts[weight], width=0.4, alpha=0.5, label='ground truth')
        plt.bar(xvals, self.pred_ind_wgts[weight], width=0.4, alpha=0.5, label='predicted')
        for metric, val in dist.items():
            plt.plot([], [], ' ', label=f'{metric}: {val:.2f}')
        plt.xlabel('Rank (among samples)')
        plt.ylabel(weight.title())
        plt.legend()
        if savefig:
            plt.savefig(os.path.join('img', f'{label}_{weight}{"_log" if ylogscale else ""}.png'))
        if show_plot:
            plt.show()

class Exp:

    def __init__(self):
        self.models = dict()
        self.model_fit_kwargs = dict()
        self.meters = defaultdict(PerformanceMeter)
        self.dataset = None
        self.last_rank_num_iter = None

    @staticmethod
    def load_exp_results(exp_res_path):
        d = pickle.load(open(exp_res_path, 'rb'))
        exp = Exp()
        exp.dname = d['dname']
        if d['exp_type'] == 'ranking':
            meters = d['meter_dict']
            exp.meters = dict((name,PerformanceMeter(meter)) for name, meter in meters.items())
            exp.last_rank_num_iter = d['last_rank_num_iter']
        return exp

    def set_data(self, dname):
        self.dname = dname
        self.dataset = ExpDataset(dname)

    def set_model(self, model_name, model, fit_kwargs={}):
        self.models[model_name] = model
        self.model_fit_kwargs[ model_name] = fit_kwargs

    def initialize_models(self):
        for name, model in self.models.items():
            model.initialize_model()

    def fit_models_to_sample(self, sampled_resps):
        for model_name, model in self.models.items():
            fit_kwargs = self.model_fit_kwargs[model_name]
            model.fit(sampled_resps, **fit_kwargs)

    def run_exp_ranking(self, sample_size=100, num_iter=100, out_name=None, save_scores_only=False):
        def _save_meters(out_name):
            if out_name is not None:
                meter_dict = to_dict(self.meters)
                out_dict = {
                    'exp_type': 'ranking',
                    'dname': self.dname,
                    'meter_dict': meter_dict,
                    'last_rank_num_iter': self.last_rank_num_iter
                }
                out_name = os.path.join(OUT_DIR, out_name + '.pkl')
                pickle.dump(out_dict, open(out_name, 'wb'))

        def _save_scores(all_iters, out_name):
            if out_name is not None:
                out_dict = {
                    'last_rank_num_iter': self.last_rank_num_iter,
                    'values': all_iters
                }
                out_name = os.path.join(OUT_DIR, out_name + '.scores.pkl')
                pickle.dump(out_dict, open(out_name, 'wb'))

        all_iters = []
        self.last_rank_num_iter = num_iter
        try:
            for i in trange(num_iter):
                sampled_resps = self.dataset.get_random_samples(sample_size)

                self.initialize_models()
                self.fit_models_to_sample(sampled_resps)
                evaluator = Evaluator(self.dataset, sampled_resps)

                singleton_idxs = sampled_resps['unique_idxs'][evaluator.singleton_locs]
                curr_iter = {
                    'singleton_idxs': singleton_idxs,
                    'scores': {}
                }
                for model_name, model in self.models.items():
                    scores = model.get_and_reset_scores()
                    for scorer_name, (ordered_idxs, ordered_scores) in scores.items():
                        config_name =  f'{model_name}_{scorer_name}'

                        curr_iter['scores'][config_name] = (ordered_idxs, ordered_scores)
                        if save_scores_only:
                            continue

                        evaluator.induce_dist_and_fix_order(ordered_idxs, ordered_scores)
                        rank_perf = evaluator.get_ranking_metric()
                        self.meters[config_name].add_perf('rank_pred', rank_perf)

                        curr_iter['scores'][config_name] = (ordered_idxs, ordered_scores)
                all_iters.append(curr_iter)

            _save_scores(all_iters, out_name)
            if not save_scores_only:
                _save_meters(out_name)
        except KeyboardInterrupt:
            print('handling exception... saving meters')
            _save_meters(out_name)
            traceback.print_exc()
