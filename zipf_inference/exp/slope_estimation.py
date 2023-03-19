from zipf_inference.exp import Exp
from zipf_inference.scorer import score_frac_singleton
from zipf_inference.dataset import NL_DESC
from zipf_inference.models import (
    NullModel, PreComputedEmbsModel, TokenBasedModel, CodeBERT, CuBERT
)
from zipf_inference.scorer import (
    GeneralizedKNNScorer, VariableKDEScorer,
    CosineSimScorer, EditDistanceScorer, KDEScorer, DETreeScorer
)

import argparse
import logging

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--out-name', type=str, required=True)
    parser.add_argument('--device', type=int, default=None)
    parser.add_argument('--num-iter', type=int, default=700)
    args = parser.parse_args()
    return args

if __name__=='__main__':
    logging.basicConfig(level=logging.ERROR)

    args = parse_args()
    device = f'cuda:{args.device}' if args.device is not None else 'cpu'

    dist_scorer_configs = dict(
        knnde_gauss_unit_k5=dict(
            scorer_class=GeneralizedKNNScorer,
            func_type='gaussian',
            call_kwargs=dict(
                k=5,
                unit_norm=True
            )
        ),
        knnde_lin_unit_k5=dict(
            scorer_class=GeneralizedKNNScorer,
            func_type='linear',
            call_kwargs=dict(
                k=5,
                unit_norm=True
            )
        ),
        knnde_gauss_k5=dict(
            scorer_class=GeneralizedKNNScorer,
            func_type='gaussian',
            call_kwargs=dict(
                k=5,
                unit_norm=False
            )
        ),
        knnde_lin_k5=dict(
            scorer_class=GeneralizedKNNScorer,
            func_type='linear',
            call_kwargs=dict(
                k=5,
                unit_norm=False
            )
        ),
        knnde_gauss_unit_k10=dict(
            scorer_class=GeneralizedKNNScorer,
            func_type='gaussian',
            call_kwargs=dict(
                k=10,
                unit_norm=True
            )
        ),
        knnde_lin_unit_k10=dict(
            scorer_class=GeneralizedKNNScorer,
            func_type='linear',
            call_kwargs=dict(
                k=10,
                unit_norm=True
            )
        ),
        knnde_gauss_k10=dict(
            scorer_class=GeneralizedKNNScorer,
            func_type='gaussian',
            call_kwargs=dict(
                k=10,
                unit_norm=False
            )
        ),
        knnde_lin_k10=dict(
            scorer_class=GeneralizedKNNScorer,
            func_type='linear',
            call_kwargs=dict(
                k=10,
                unit_norm=False
            )
        ),

        var_kde_gauss_unit_k5=dict(
            scorer_class=VariableKDEScorer,
            func_type='gaussian',
            call_kwargs=dict(
                unit_norm=True,
                k=5,
            )
        ),
        var_kde_gauss_unit_k10=dict(
            scorer_class=VariableKDEScorer,
            func_type='gaussian',
            call_kwargs=dict(
                unit_norm=True,
                k=10,
            )
        ),
        var_kde_gauss_k5=dict(
            scorer_class=VariableKDEScorer,
            func_type='gaussian',
            call_kwargs=dict(
                k=5,
            )
        ),
        var_kde_gauss_k10=dict(
            scorer_class=VariableKDEScorer,
            func_type='gaussian',
            call_kwargs=dict(
                k=10,
            )
        ),
        var_kde_linear_unit_k5=dict(
            scorer_class=VariableKDEScorer,
            func_type='linear',
            call_kwargs=dict(
                unit_norm=True,
                k=5,
            )
        ),
        var_kde_linear_unit_k10=dict(
            scorer_class=VariableKDEScorer,
            func_type='linear',
            call_kwargs=dict(
                unit_norm=True,
                k=10,
            )
        ),
        var_kde_linear_k5=dict(
            scorer_class=VariableKDEScorer,
            func_type='linear',
            call_kwargs=dict(
                k=5,
            )
        ),
        var_kde_linear_k10=dict(
            scorer_class=VariableKDEScorer,
            func_type='linear',
            call_kwargs=dict(
                k=10,
            )
        ),
    )

    exp = Exp()
    exp.set_data(args.data)
    nl_desc = NL_DESC[args.data]

    tokenmodel = TokenBasedModel(device=device, exp=exp)
    tokenmodel.set_many_scorers(dist_scorer_configs)
    tokenmodel.set_scorer(
        scorer_class=EditDistanceScorer,
        func_type='freq_topk_sum',
        name='ed_top10',
        call_kwargs={'k': 10}
    )
    tokenmodel.set_scorer(
        scorer_class=EditDistanceScorer,
        func_type='freq_topk_sum',
        name='ed_top5',
        call_kwargs={'k': 5}
    )
    tokenmodel.set_scorer(
        scorer_class=EditDistanceScorer,
        func_type='freq_topk_sum',
        name='ed_all',
        call_kwargs={
            'k': None
        }
    )
    exp.set_model('token', tokenmodel)

    null = NullModel(
        device=device,
        exp=exp)
    null.set_scorer(
        name='frac_singleton',
        scorer_func=score_frac_singleton)

    exp.initialize_models()
    exp.run_exp_ranking(sample_size=70,
                        num_iter=args.num_iter,
                        out_name=args.out_name,
                        save_scores_only=True)
