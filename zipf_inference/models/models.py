from transformers import BertTokenizer, BertModel
from transformers import GPT2Tokenizer, GPT2Model
from transformers import RobertaTokenizer, RobertaModel

from tensor2tensor.data_generators import text_encoder
from cubert import python_tokenizer
from cubert.code_to_subtokenized_sentences import code_to_cubert_sentences
from tqdm.autonotebook import tqdm, trange
import itertools
import os

from PIL import Image
import numpy as np
import torch
from torchvision import transforms as T

VOCAB_FILE = os.path.join('cubert', 'bin', 'github_python_minus_ethpy150open_deduplicated_vocabulary.txt')
CHECKPOINT_PATH = 'cubert/bin/'

class CodeBERT:

    def __init__(self, device='cuda:0', mode='eval'):
        self.device = device
        self.tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
        self.model = RobertaModel.from_pretrained("microsoft/codebert-base").to(device)
        if mode == 'eval':
            self.model = self.model.eval()

    def convert_tokens_to_ids(self, nl_descs, ranked_progs):
        nl_tokens = self.tokenizer.tokenize(nl_descs)
        self.model.config.max_position_embeddings
        cl_tokens_max_len = 512 - (len(nl_tokens) + 3)
        tokens_lst = []
        for prog in tqdm(ranked_progs):
            cl_tokens = self.tokenizer.tokenize(" ".join(prog.split()))
            tokens = ([self.tokenizer.cls_token] + nl_tokens + [self.tokenizer.sep_token] +
                      cl_tokens[:cl_tokens_max_len] + [self.tokenizer.sep_token])
            tokens_lst.append(tokens)
        max_len = max(len(tokens) for tokens in tokens_lst)

        tokens_lst = [tokens + [self.tokenizer.pad_token]*(max_len - len(tokens)) for tokens in tokens_lst]
        token_ids_lst = [self.tokenizer.convert_tokens_to_ids(tokens) for tokens in tokens_lst]
        return token_ids_lst

    def compute_embeddings(self, token_ids_lst, batch_size=200):
        embs = []
        with torch.no_grad():
            steps = len(token_ids_lst)//batch_size + 1
            for i in trange(steps):
                inp = torch.tensor(token_ids_lst[200*i:200*(i+1)]).to(self.device)
                context_embeddings = self.model(inp)
                del inp
                embs.append(context_embeddings.pooler_output)
        cat_embs = torch.cat(embs)
        torch.cuda.empty_cache()
        return cat_embs

class CuBERT:

    def __init__(self, device='cuda:0', mode='eval'):
        self.bert_tokenizer = BertTokenizer(VOCAB_FILE)
        self.model = BertModel.from_pretrained(pretrained_model_name_or_path=CHECKPOINT_PATH).to(device)
        if mode == 'eval':
            self.model = self.model.eval()
        self.subword_tokenizer = text_encoder.SubwordTextEncoder(VOCAB_FILE)
        self.code_tokenizer = python_tokenizer.PythonTokenizer()
        self.device = device

    def convert_tokens_to_ids(self, nl_desc, ranked_progs, return_raw=False):
        tokens_lst = []
        for prog in tqdm(ranked_progs, position=0, leave=True):
            prog = f'"""\n{nl_desc}\n"""\n' + prog
            sent = code_to_cubert_sentences(prog, self.code_tokenizer, self.subword_tokenizer)
            sent_flat = list(itertools.chain(*sent))[:self.model.config.max_position_embeddings - 2]
            sent_flat = [f"'{tok}'" for tok in sent_flat]
            sent_toks = (["'[CLS]_'"] + sent_flat + ["'[SEP]_'"])
            tokens_lst.append(sent_toks)
        if return_raw:
            return tokens_lst
        max_len = max(len(toks) for toks in tokens_lst)
        tokens_lst = [toks + ["'<pad>_'"]*(max_len - len(toks)) for toks in tokens_lst]
        token_ids_lst = [self.bert_tokenizer.convert_tokens_to_ids(tokens) for tokens in tqdm(tokens_lst)]
        return token_ids_lst

    def compute_embeddings(self, token_ids_lst, batch_size=200):
        embs = []
        with torch.no_grad():
            steps = len(token_ids_lst)//batch_size + 1
            for i in trange(steps):
                inp = torch.tensor(token_ids_lst[200*i:200*(i+1)]).to(self.device)
                context_embeddings = self.model(inp)
                del inp
                embs.append(context_embeddings.pooler_output)
        cat_embs = torch.cat(embs)
        torch.cuda.empty_cache()
        return cat_embs

