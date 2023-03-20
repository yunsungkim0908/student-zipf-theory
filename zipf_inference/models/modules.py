import torch
from torch import nn
import torch.nn.utils.rnn as rnn_utils

class GRUEncoder(nn.Module):

    def __init__(self,
                 vocab,
                 hidden_dim,
                 output_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        vocab_size = len(vocab)
        self.embeddings = nn.Embedding(vocab_size, hidden_dim)
        self.encoder = nn.GRU(input_size=hidden_dim,
                               hidden_size=hidden_dim,
                               num_layers=1,
                               batch_first=True,
                               bidirectional=True)
        self.h_map = nn.Linear(2*hidden_dim, output_dim)

    def forward(self, src_batch):
        token_ids = src_batch['token_ids']
        length = src_batch['length']
        embs = self.embeddings(token_ids)
        embs = rnn_utils.pack_padded_sequence(embs, lengths=length, batch_first=True, enforce_sorted=False)
        enc, h = self.encoder(embs)
        h = self.h_map(torch.cat([h[0], h[1]], dim=1).view(-1, 2*self.hidden_dim))
        return h

class GRUDecoder(nn.Module):

    def __init__(self,
                 vocab,
                 input_dim,
                 hidden_dim,
                 max_len,):
        super().__init__()
        self.max_len = max_len
        self.input_dim = input_dim

        self.vocab = vocab
        vocab_size = len(vocab)
        self.embedding = nn.Embedding(vocab_size, embedding_dim=input_dim)
        self.rnn = nn.GRU(input_size=input_dim,
                           hidden_size=hidden_dim,
                           num_layers=1,
                           batch_first=True)
        self.ffn = nn.Sequential(
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, vocab_size)
        )

    def forward(self, src_hidden, tgt_batch, diff=None):
        token_ids = tgt_batch['token_ids']
        batch_size, seq_len = token_ids.size()
        tgt_embs = self.embedding(token_ids)
        if diff is not None:
            diff = diff.unsqueeze(1).repeat(1, seq_len, 1)
            inp = torch.cat([tgt_embs, diff], dim=2)
        else:
            inp = tgt_embs
        out, _ = self.rnn(inp, src_hidden.unsqueeze(0))
        out_size = out.size()
        out = self.ffn(out.contiguous().view(-1, out_size[-1]))
        return out.view(*out_size[:-1], -1)

    # No need for eval_forward. Not interested in the actual token output
