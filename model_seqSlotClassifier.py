from typing import Dict

import torch
import torch.nn as nn
from torch.nn import Embedding
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import math

class SeqSlotClassifier(torch.nn.Module):
    def __init__(
        self,
        embeddings: torch.tensor,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        num_class: int,
    ) -> None:
        super(SeqSlotClassifier, self).__init__()
        self.embed = Embedding.from_pretrained(embeddings, freeze=False)
        # TODO: model architecture
        self.embed_dim = embeddings.size(1)
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.num_class = num_class
        self.gru = nn.GRU(self.embed_dim, self.hidden_size, num_layers = self.num_layers, dropout = dropout,
         batch_first = True, bidirectional = self.bidirectional)
        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.out = nn.Linear(hidden_size // 2, num_class)
        self.act_fn = nn.LeakyReLU(0.1)
        self.act_fnP = nn.PReLU()
        self.layer_1 = nn.Linear(hidden_size * 2, hidden_size)
        self.layer_2 = nn.Linear(hidden_size, hidden_size // 2)
        
    """
    @property
    def encoder_output_size(self) -> int:
        # TODO: calculate the output dimension of rnn
        raise NotImplementedError
    """
    
    def forward(self, batch, h) -> Dict[str, torch.Tensor]:
        # TODO: implement model forward
        h_embedding = self.embed(batch)
        h_embedding = torch.squeeze(torch.unsqueeze(h_embedding, 0))
        h_embedding = self.dropout(h_embedding)
        out, h = self.gru(h_embedding, h)
        #print(out.shape)
        out = self.dropout(out)
        out = self.act_fn(out)
        out = self.layer_1(out)
        out = self.dropout(out)
        out = self.act_fn(out)
        out = self.layer_2(out)
        out = self.dropout(out)
        out = self.act_fn(out)
        out = self.out(out)
        #print(out.shape)
        return out, h

    def init_hidden(self, batch_size, device):
        return torch.autograd.Variable(torch.zeros(2 * self.num_layers, batch_size, self.hidden_size)).to(device)

class SeqSlotLSTMClassifier(torch.nn.Module):
    def __init__(
        self,
        embeddings: torch.tensor,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        num_class: int,
    ) -> None:
        super(SeqSlotLSTMClassifier, self).__init__()
        self.embed = Embedding.from_pretrained(embeddings, freeze=False)
        # TODO: model architecture
        self.embed_dim = embeddings.size(1)
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.num_class = num_class
        self.dropout = dropout
        self.lstm = nn.LSTM(self.embed_dim, self.hidden_size, num_layers = self.num_layers, dropout = self.dropout,
         batch_first = True, bidirectional = self.bidirectional)
        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.out = nn.Linear(hidden_size//2, num_class)
        self.act_fn = nn.LeakyReLU(0.1)
        self.act_fnP = nn.PReLU()
        self.layer_1 = nn.Linear(hidden_size * 2, hidden_size)
        self.layer_2 = nn.Linear(hidden_size, hidden_size // 2)
    """    
    @property
    def encoder_output_size(self) -> int:
        # TODO: calculate the output dimension of rnn
        raise NotImplementedError
    """

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        # TODO: implement model forward
        #print(batch.shape)
        h_embedding = self.embed(batch)
        h_embedding = torch.squeeze(torch.unsqueeze(h_embedding, 0))
        #print(h_embedding.shape)
        h_embedding = self.dropout(h_embedding)
        lstm_out, (ht, ct) = self.lstm(h_embedding)
        #ht = torch.cat((ht[-1], ht[-2]), axis=-1)
        #print(lstm_out.shape)
        out = self.dropout(lstm_out)
        out = self.act_fn(out)
        out = self.layer_1(out)
        out = self.dropout(out)
        out = self.act_fn(out)
        out = self.layer_2(out)
        out = self.dropout(out)
        out = self.act_fn(out)
        out = self.out(out)
        #print(out.shape)
        return out

    def init_hidden(self, batch_size, device):
        return torch.autograd.Variable(torch.zeros(2 * self.num_layers, batch_size, self.hidden_size)).to(device)