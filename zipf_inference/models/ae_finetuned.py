from zipf_inference.globals import MODEL_DIR
from zipf_inference.models.modules import GRUDecoder
from zipf_inference.models.base import BaseModel
from zipf_inference.dataset.utils import batch_to_device
from zipf_inference.utils import clear_memory

from torch.utils.data import DataLoader
from tensor2tensor.data_generators import text_encoder
from .cubert import python_tokenizer
from .cubert.code_to_subtokenized_sentences import code_to_cubert_sentences

from transformers import RobertaTokenizer, RobertaModel
from transformers import BertTokenizer, BertModel
from tqdm.autonotebook import trange, tqdm
from zipf_inference.utils import check_nan, normalize_embs

import os
import itertools
import torch
import torch.nn as nn
import torch.optim as optim

class AutoEncFinetuned(BaseModel):

    def __init__(self, input_dim, max_len, emb_dim=None, norm_emb=False, **kwargs):
        BaseModel.__init__(self, **kwargs)
        self.input_dim = input_dim
        self.max_len = max_len
        self.emb_dim = emb_dim
        self.norm_emb = norm_emb
        self.no_ffn = emb_dim is None

    def initialize_model(self):
        super().initialize_model()

        self.initialize_pretrained()
        self.emb_dim = self.raw_enc_dim if self.emb_dim is None else self.emb_dim
        if not self.no_ffn:
            self.ffn = nn.Sequential(
                nn.Linear(self.raw_enc_dim, self.raw_enc_dim // 2),
                nn.BatchNorm1d(self.raw_enc_dim // 2),
                nn.GELU(),
                nn.Linear(self.raw_enc_dim // 2, self.emb_dim),
                nn.BatchNorm1d(self.emb_dim),
                nn.GELU()
            )
        self.decoder = GRUDecoder(self.exp_dataset.vocab, self.input_dim, self.emb_dim, self.max_len)
        self.to(self.device)

    def initialize_pretrained(self):
        raise NotImplementedError

    def get_loss(self, batch, **enc_kwargs):
        latent = self.encode(batch, **enc_kwargs)
        out = self.decoder(latent, batch['tgt'])[:,1:]

        tgt_ids = batch['tgt']['token_ids'][:,:-1]
        tgt_len = [length-1 for length in batch['tgt']['length']]
        mask = [[1]*length + [0]*(max(tgt_len)-length) for length in tgt_len]
        mask = torch.BoolTensor(mask).to(self.device)

        masked_out_flat = (torch.masked_select(out, mask.unsqueeze(2))
                                .view(-1, out.size(-1)))
        masked_tgt_flat = torch.masked_select(tgt_ids, mask)

        criterion = nn.CrossEntropyLoss()
        loss = criterion(masked_out_flat, masked_tgt_flat)
        return loss

    def fit(self, sampled_resps, epochs=10, verbose=False, lr=3e-3, eval_epochs=[], **enc_kwargs):

        def _eval(label, epoch): # epoch is num epochs completed
            if epoch not in eval_epochs and epoch != 0:
                return
            self.eval()
            with torch.no_grad():
                eval_loader = DataLoader(model_dataset, batch_size=200, shuffle=False,
                                    collate_fn=model_dataset.collate_fn)
                embs_lst = []
                for batch in eval_loader:
                    batch = batch_to_device(self.device, batch)
                    embs = self.encode(batch, **enc_kwargs)
                    embs_lst.append(embs)
                self.embs = torch.cat(embs_lst)

            for scorer_name in self.scorers.keys():
                self.call_scorer(scorer_name, sampled_resps, call_label=label)

        model_dataset = sampled_resps['model_dataset']
        train_loader = DataLoader(model_dataset, batch_size=200, shuffle=True,
                            collate_fn=model_dataset.collate_fn)

        optimizer = optim.Adam(self.parameters(), lr=lr)

        _eval('ep0', epoch=0)

        for epoch in trange(epochs, disable=not verbose):
            self.train()
            for batch in train_loader:
                batch = batch_to_device(self.device, batch)
                optimizer.zero_grad()
                loss = self.get_loss(batch, **enc_kwargs)
                loss.backward()
                optimizer.step()
                if verbose:
                    tqdm.write(f'iter {epoch}: loss {loss.data}')

            _eval(f'ep{epoch+1}', epoch=epoch+1)

    def encode(self, **enc_kwargs):
        raise NotImplementedError

class AutoEncCodeBERT(AutoEncFinetuned):

    def initialize_pretrained(self):
        self.tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
        self.encoder = RobertaModel.from_pretrained("microsoft/codebert-base")
        self.encoder.embeddings.requires_grad_(False)
        for i in range(11):
            self.encoder.encoder.layer[i].requires_grad_(False)
        self.raw_enc_dim = self.encoder.config.hidden_size

    def encode(self, batch, nl_desc):
        nl_tokens = self.tokenizer.tokenize(nl_desc)
        self.encoder.config.max_position_embeddings
        cl_tokens_max_len = 150 - (len(nl_tokens)+3)
        # cl_tokens_max_len = 512 - (len(nl_tokens) + 3)
        tokens_lst = []
        for inp in batch['src']['raw_inp']:
            cl_tokens = self.tokenizer.tokenize(" ".join(inp.split()))
            tokens = ([self.tokenizer.cls_token] + nl_tokens + [self.tokenizer.sep_token] +
                      cl_tokens[:cl_tokens_max_len] + [self.tokenizer.sep_token])
            tokens_lst.append(tokens)
        max_len = max(len(tokens) for tokens in tokens_lst)

        tokens_lst = [tokens + [self.tokenizer.pad_token]*(max_len - len(tokens)) for tokens in tokens_lst]
        batch_tok_ids = [self.tokenizer.convert_tokens_to_ids(tokens) for tokens in tokens_lst]
        inp_tensor = torch.tensor(batch_tok_ids).to(self.device)

        raw_enc = self.encoder(inp_tensor).pooler_output
        out = self.ffn(raw_enc) if not self.no_ffn else raw_enc
        del inp_tensor
        clear_memory()
        if self.norm_emb:
            out = normalize_embs(out)
        clear_memory()
        return out

class AutoEncCuBERT(AutoEncFinetuned):

    def initialize_pretrained(self):
        vocab_file = os.path.join(MODEL_DIR, 'cubert', 'bin',
                  'github_python_minus_ethpy150open_deduplicated_vocabulary.txt')
        checkpoint_path = os.path.join(MODEL_DIR, 'cubert', 'bin')
        self.bert_tokenizer = BertTokenizer(vocab_file)
        self.encoder = BertModel.from_pretrained(pretrained_model_name_or_path=checkpoint_path)
        self.encoder.embeddings.requires_grad_(False)
        for i in range(11):
            self.encoder.encoder.layer[i].requires_grad_(False)
        self.subword_tokenizer = text_encoder.SubwordTextEncoder(vocab_file)
        self.code_tokenizer = python_tokenizer.PythonTokenizer()
        self.raw_enc_dim = self.encoder.config.hidden_size

    def encode(self, batch, nl_desc):
        tokens_lst = []
        cl_tokens_max_len = 150
        # cl_tokens_max_len = self.encoder.config.max_position_embeddings - 2
        for inp in batch['src']['raw_inp']:
            inp = f'"""\n{nl_desc}\n"""\n' + inp
            sent = code_to_cubert_sentences(inp, self.code_tokenizer, self.subword_tokenizer)
            sent_flat = list(itertools.chain(*sent))[:cl_tokens_max_len]
            sent_flat = [f"'{tok}'" for tok in sent_flat]
            sent_toks = (["'[CLS]_'"] + sent_flat + ["'[SEP]_'"])
            tokens_lst.append(sent_toks)
        max_len = max(len(toks) for toks in tokens_lst)
        tokens_lst = [toks + ["'<pad>_'"]*(max_len - len(toks)) for toks in tokens_lst]
        inputs = [self.bert_tokenizer.convert_tokens_to_ids(tokens) for tokens in tokens_lst]

        inp_tensor = torch.tensor(inputs).to(self.device)

        raw_enc = self.encoder(inp_tensor).pooler_output
        out = self.ffn(raw_enc) if not self.no_ffn else raw_enc
        del inp_tensor
        clear_memory()
        if self.norm_emb:
            out = normalize_embs(out)
        clear_memory()
        return out

