from zipf_inference.exp import Exp
from zipf_inference.scorer import score_frequency, score_freq_head_rand_tail
from zipf_inference.dataset import NL_DESC
from zipf_inference.models import (
    NullModel, PreComputedEmbsModel, TokenBasedModel, CodeBERT, CuBERT
)
from zipf_inference.scorer import (
    GeneralizedKNNScorer, VariableKDEScorer, TokenBasedScorer,
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
    parser.add_argument('--scores-only', action="store_true", default=False)
    args = parser.parse_args()
    return args

if __name__=='__main__':
    logging.basicConfig(level=logging.ERROR)

    args = parse_args()
    device = f'cuda:{args.device}' if args.device is not None else 'cpu'

    cosine_sim_configs = dict(
        hyb=dict(
            scorer_class=CosineSimScorer,
            func_type='freq_topk_sum'
        ),
        hyb_abs=dict(
            scorer_class=CosineSimScorer,
            func_type='freq_topk_sum',
            call_kwargs={
                'absolute': True
            }
        ),
        hyb_mul=dict(
            scorer_class=CosineSimScorer,
            func_type='freq_topk_sum',
            call_kwargs={
                'multiplicity': True
            }
        ),
        hyb_abs_mul=dict(
            scorer_class=CosineSimScorer,
            func_type='freq_topk_sum',
            call_kwargs={
                'absolute': True,
                'multiplicity': True
            }
        ),
    )

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

    embs_only_scorer_configs=dict(
        kde_linear_unit=dict(
            scorer_class=KDEScorer,
            func_type='linear',
            call_kwargs=dict(
                unit_norm=True,
            )
        ),
        kde_linear=dict(
            scorer_class=KDEScorer,
            func_type='linear',
        ),
        kde_gauss_unit=dict(
            scorer_class=KDEScorer,
            func_type='gaussian',
            call_kwargs=dict(
                unit_norm=True,
            )
        ),
        kde_gauss=dict(
            scorer_class=KDEScorer,
            func_type='gaussian',
        ),
        det=dict(
            scorer_class=DETreeScorer,
        ),
    )

    exp = Exp()
    exp.set_data(args.data)
    nl_desc = NL_DESC[args.data]

    cubert = PreComputedEmbsModel(
        embs_fname=f'cu_{args.data}',
        norm_emb=True,
        device=device,
        exp=exp
    )
    cubert.set_many_scorers(cosine_sim_configs)
    cubert.set_many_scorers(dist_scorer_configs)
    cubert.set_many_scorers(embs_only_scorer_configs)
    exp.set_model('cubert', cubert)

    codebert = PreComputedEmbsModel(
        embs_fname=f'co_{args.data}',
        norm_emb=True,
        device=device,
        exp=exp
    )
    codebert.set_many_scorers(cosine_sim_configs)
    codebert.set_many_scorers(dist_scorer_configs)
    codebert.set_many_scorers(embs_only_scorer_configs)
    exp.set_model('codebert', codebert)

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
    tokenmodel.set_scorer(
        scorer_class=TokenBasedScorer,
        func_type='length',
        name='length',
    )
    tokenmodel.set_scorer(
        scorer_class=TokenBasedScorer,
        func_type='length_diff_from_mean',
        name='length_diff',
    )

    exp.set_model('token', tokenmodel)

    null = NullModel(
        device=device,
        exp=exp)
    null.set_scorer(
        name='frac_singleton',
        scorer_func=score_frac_singleton)
    null.set_scorer(
        name='freq',
        scorer_func=score_frequency)
    null.set_scorer(
        name='freq_rand',
        scorer_func=score_freq_head_rand_tail)
    exp.set_model('base', null)

    exp.initialize_models()
    exp.run_exp_ranking(sample_size=70,
                        num_iter=args.num_iter,
                        out_name=args.out_name,
                        save_scores_only=args.scores_only)
