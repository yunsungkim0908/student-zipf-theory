from zipf_inference.exp import Exp
from zipf_inference.scorer import score_frac_singleton
from zipf_inference.models import NullModel
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

    exp = Exp()
    exp.set_data(args.data)

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
