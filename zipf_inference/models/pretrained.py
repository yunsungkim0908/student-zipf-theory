from .base import PretrainedModel
from zipf_inference.globals import MODEL_DIR

from transformers import BertTokenizer, BertModel
from transformers import GPT2Tokenizer, GPT2Model
from transformers import RobertaTokenizer, RobertaModel

from tensor2tensor.data_generators import text_encoder
from .cubert import python_tokenizer
from .cubert.code_to_subtokenized_sentences import code_to_cubert_sentences
from tqdm.autonotebook import tqdm, trange
import itertools
import os

from PIL import Image
import numpy as np
import torch
from torchvision import transforms as T

class CodeBERT(PretrainedModel):

    def __init__(self, nl_desc, **kwargs):
        self.nl_desc = nl_desc
        super().__init__(**kwargs)

    def setup_pretrained(self):
        self.tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
        self.model = RobertaModel.from_pretrained("microsoft/codebert-base").to(self.device)

    def convert_all_inputs_to_tensors(self):
        nl_tokens = self.tokenizer.tokenize(self.nl_desc)
        self.model.config.max_position_embeddings
        # cl_tokens_max_len = 150 - (len(nl_tokens) + 3)
        cl_tokens_max_len = 512 - (len(nl_tokens) + 3)
        tokens_lst = []
        for prog in tqdm(self.exp_dataset.ranked_resps):
            cl_tokens = self.tokenizer.tokenize(" ".join(prog.split()))
            tokens = ([self.tokenizer.cls_token] + nl_tokens + [self.tokenizer.sep_token] +
                      cl_tokens[:cl_tokens_max_len] + [self.tokenizer.sep_token])
            tokens_lst.append(tokens)
        max_len = max(len(tokens) for tokens in tokens_lst)

        tokens_lst = [tokens + [self.tokenizer.pad_token]*(max_len - len(tokens)) for tokens in tokens_lst]
        token_ids_lst = [self.tokenizer.convert_tokens_to_ids(tokens) for tokens in tokens_lst]
        return token_ids_lst

    def compute_all_embeddings(self, token_ids_lst, batch_size=200):
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

class CuBERT(PretrainedModel):

    def __init__(self, nl_desc, **kwargs):
        self.nl_desc = nl_desc
        super().__init__(**kwargs)

    def setup_pretrained(self):
        vocab_file = os.path.join(MODEL_DIR, 'cubert', 'bin',
                  'github_python_minus_ethpy150open_deduplicated_vocabulary.txt')
        checkpoint_path = os.path.join(MODEL_DIR, 'cubert', 'bin')
        self.bert_tokenizer = BertTokenizer(vocab_file)
        self.model = BertModel.from_pretrained(pretrained_model_name_or_path=checkpoint_path).to(self.device)
        self.subword_tokenizer = text_encoder.SubwordTextEncoder(vocab_file)
        self.code_tokenizer = python_tokenizer.PythonTokenizer()

    def convert_all_inputs_to_tensors(self):
        tokens_lst = []
        # cl_tokens_max_len = 150
        cl_tokens_max_len = self.model.config.max_position_embeddings - 2
        for prog in tqdm(self.exp_dataset.ranked_resps, position=0, leave=True):
            prog = f'"""\n{self.nl_desc}\n"""\n' + prog
            sent = code_to_cubert_sentences(prog, self.code_tokenizer, self.subword_tokenizer)
            sent_flat = list(itertools.chain(*sent))[:cl_tokens_max_len]
            sent_flat = [f"'{tok}'" for tok in sent_flat]
            sent_toks = (["'[CLS]_'"] + sent_flat + ["'[SEP]_'"])
            tokens_lst.append(sent_toks)
        max_len = max(len(toks) for toks in tokens_lst)
        tokens_lst = [toks + ["'<pad>_'"]*(max_len - len(toks)) for toks in tokens_lst]
        inputs = [self.bert_tokenizer.convert_tokens_to_ids(tokens) for tokens in tqdm(tokens_lst)]
        return inputs

    def compute_all_embeddings(self, inputs, batch_size=200):
        embs = []
        with torch.no_grad():
            steps = len(inputs)//batch_size + 1
            for i in trange(steps):
                inp = torch.tensor(inputs[200*i:200*(i+1)]).to(self.device)
                context_embeddings = self.model(inp)
                del inp
                embs.append(context_embeddings.pooler_output)
        cat_embs = torch.cat(embs)
        torch.cuda.empty_cache()
        return cat_embs

class ResNext50(PretrainedModel):

    def __init__(self, img_dir, **kwargs):
        self.img_dir = img_dir
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'resnext50_32x4d',
                                    pretrained=True).to(self.device)
        super().__init__(self, **kwargs)

    def convert_all_inputs_to_tensors(self):
        return self.exp_dataset.ranked_resps

    def compute_all_embeddings(self, inputs, batch_size=200):
        preprocess = T.Compose([
            T.Resize((256,256)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        embs = []
        with torch.no_grad():
            steps = len(inputs) // batch_size + 1
            for i in trange(steps):
                input_paths = inputs[batch_size*i:batch_size*(i+1)]
                input_tensors = []
                for path in input_paths:
                    input_image = Image.open(os.path.join(self.img_dir, path))
                    input_image = Image.fromarray(np.array(input_image)[...,:3])
                    input_tensors.append(preprocess(input_image).to(self.device))
                inp = torch.stack(input_tensors)
                embeddings = self.model(inp)
                embs.append(embeddings)
                del inp
        embs = torch.cat(embs)
        torch.cuda.empty_cache()
        return embs

class GPT2(PretrainedModel):

    def __init__(self, **kwargs):
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.unk_token
        self.model = GPT2Model.from_pretrained('gpt2').to(self.device)
        super().__init__(self, **kwargs)

    def convert_all_inputs_to_tensors(self):
        """
            inputs: List[str]
        """
        inputs = self.exp_dataset.ranked_resps
        raw_tokens_lst = [self.tokenizer.tokenize(inp) for inp in tqdm(inputs)]
        max_len = min(512, max(len(toks) for toks in raw_tokens_lst))
        tokens_lst, masks_lst = [], []
        for tokens in raw_tokens_lst:
            pad_size = max(0, max_len - len(tokens))
            tokens = [self.tokenizer.pad_token]*pad_size + tokens[-max_len:]
            masks = [0]*pad_size + [1]*(max_len - pad_size)
            tokens_lst.append(tokens)
            masks_lst.append(masks)
        tokens_lst = [self.tokenizer.convert_tokens_to_ids(tokens) for tokens in tokens_lst]

        return dict(input_ids=tokens_lst, attention_mask=masks_lst)

    def compute_all_embeddings(self, inputs, batch_size=200):
        embs = []
        with torch.no_grad():
            steps = len(inputs['input_ids']) // batch_size + 1
            for i in trange(steps):
                input_ids = inputs['input_ids'][batch_size*i:batch_size*(i+1)]
                attention_mask = inputs['attention_mask'][batch_size*i:batch_size*(i+1)]
                input_ids = torch.LongTensor(input_ids).to(self.device)
                attention_mask = torch.FloatTensor(attention_mask).to(self.device)

                out = self.model(input_ids=input_ids, attention_mask=attention_mask)
                embeddings = out.last_hidden_state[:,-1,:].squeeze(1)
                del input_ids, attention_mask
                embs.append(embeddings)
        embs = torch.cat(embs)
        torch.cuda.empty_cache()
        return embs

