import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from zipf_inference.models.base import BaseModel
from zipf_inference.models.modules import GRUEncoder, GRUDecoder
from zipf_inference.dataset.utils import batch_to_device

class AutoEnc(BaseModel):

    def __init__(self, hidden_dim, output_dim, max_len, device, exp_dataset):
        BaseModel.__init__(self, device, exp_dataset)
        self.encoder = GRUEncoder(exp_dataset.vocab, hidden_dim, output_dim)
        self.decoder = GRUDecoder(exp_dataset.vocab, output_dim, hidden_dim, max_len)
        self.to(device)

    def get_loss(self, batch):
        latent = self.encoder(batch['src'])
        out = self.decoder(latent, batch['tgt'])[:,1:]

        tgt_ids = batch['tgt']['token_ids'][:,:-1]
        tgt_len = [length-1 for length in batch['tgt']['length']]
        mask = [[1]*length + [0]*(max(tgt_len)-length) for length in tgt_len]
        mask = torch.BoolTensor(mask).to(self.device)

        masked_out_flat = (torch.masked_select(out, mask.unsqueeze(2))
                                .view(-1, out.size(-1)))
        masked_tgt_flat = torch.masked_select(tgt_ids, mask)

        criterion = nn.CrossEntropyLoss()
        return criterion(masked_out_flat, masked_tgt_flat)


    def fit(self, model_dataset, epochs=100, verbose=False):
        # TODO: initialize parameters for each fit
        # print('TODO: Initialize parameters for every new fit')
        train_loader = DataLoader(model_dataset, batch_size=200, shuffle=True,
                            collate_fn=model_dataset.collate_fn)

        self.train()
        optimizer = optim.Adam(self.parameters(), lr=3e-3)
        for epoch in range(epochs):
            for batch in train_loader:
                batch = batch_to_device(self.device, batch)
                optimizer.zero_grad()
                loss = self.get_loss(batch)
                loss.backward()
                optimizer.step()
                if verbose:
                    print(f'iter {epoch}: loss {loss.data}')

        self.eval()
        with torch.no_grad():
            eval_loader = DataLoader(model_dataset, batch_size=200, shuffle=False,
                                collate_fn=model_dataset.collate_fn)
            embs = []
            for batch in eval_loader:
                batch = batch_to_device(self.device, batch)
                embs.append(self.encoder(batch['src']))
            self.embs = torch.cat(embs)
