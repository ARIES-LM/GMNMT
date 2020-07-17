import torch
import numpy as np
import torch.nn as nn
from model.util import Linear, clone, MultiHeadedAttention, PositionwiseFeedForward, SublayerConnection, GatedConnection

class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clone(layer, N)

    def forward(self, y, memory, src_mask, tgt_mask):
        for layer in self.layers:
            y = layer(y, memory, src_mask, tgt_mask)
        return y

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_hidden, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadedAttention(n_heads, d_model)
        self.src_attn = MultiHeadedAttention(n_heads, d_model)
        self.ffn = PositionwiseFeedForward(d_model, d_hidden)
        self.size = d_model
        self.sublayer = clone(SublayerConnection(d_model, dropout), 3)

    def forward(self, y, x, src_mask, tgt_mask):
        y = self.sublayer[0](y, lambda y: self.self_attn(y, y, y, tgt_mask))
        y = self.sublayer[1](y, lambda y: self.src_attn(y, x, x, src_mask))
        y = self.sublayer[2](y, self.ffn)
        return y

    def search(self, y, previous_y, x, src_mask):
        y = self.sublayer[0](y, lambda y: self.self_attn(y, previous_y, previous_y))
        y = self.sublayer[1](y, lambda y: self.src_attn(y, x, x, src_mask))
        y = self.sublayer[2](y, self.ffn)
        return y

