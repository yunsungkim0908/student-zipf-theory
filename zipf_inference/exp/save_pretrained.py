from zipf_inference.dataset import ExpDataset
from zipf_inference.dataset import NL_DESC
from zipf_inference.models import CodeBERT, CuBERT

import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()
    return args

if __name__=='__main__':
    args = parse_args()
    device = f'cuda:{args.device}'

    for dname, nl_desc in NL_DESC.items():
        print(f'dname: {dname}')
        dataset = ExpDataset(dname)

        cubert = CuBERT(
            device=device,
            exp_dataset=dataset,
            nl_desc=nl_desc)
        cubert.save_embeddings(f'cu_{dname}')

        codebert = CodeBERT(
            device=device,
            exp_dataset=dataset,
            nl_desc=nl_desc)
        codebert.save_embeddings(f'co_{dname}')

