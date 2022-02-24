import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from components.transformerNet.attention_module import *
from components.netUtils import NetUtil, CPointNet


class TransformerEncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, encoder, decoder, linear_layer):
        super(TransformerEncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.decode_linear_layer = linear_layer

    # here is a little difference in comparision to the original paper,
    # here we use the target the input to predict instead of using the previous output of decoder
    def forward(self, src, target):
        "Take in and process masked src and target sequences."
        output_encode = self.forward_encode(src)
        output_decode = self.forward_decode(output_encode, target)
        return output_encode, output_decode

    def forward_encode(self, src):
        return self.encoder(src)

    def forward_decode(self, output_encode, target):
        output_decode = self.decoder(target, output_encode)
        return self.decode_linear_layer(output_decode)


class ResidualConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, layernorm_size, dropout):
        super(ResidualConnection, self).__init__()
        self.norm = nn.LayerNorm([layernorm_size])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        # return x + F.relu(self.norm((self.dropout(sublayer(x)))))
        return x + F.relu(self.norm((sublayer(x))))
        # return self.norm(x + self.dropout(sublayer(x)))
        # return self.norm(x + self.dropout(sublayer(x)))

        ## Encoder


class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, input_linear_transform, encode_layer, N, use_cat=True):
        super(Encoder, self).__init__()
        self.input_linear_transform = input_linear_transform
        self.layers = clones(encode_layer, N)
        self.use_cat = use_cat
        self.N = N
        if use_cat:
            self.last_layer_size = encode_layer.d_model * self.N
        else:
            self.last_layer_size = encode_layer.d_model
        self.norm = nn.LayerNorm([self.last_layer_size])

    def forward(self, x):
        x = self.input_linear_transform(x)
        ori_x = x
        hidden_state = []
        for layer in self.layers:
            x = layer(x)
            hidden_state.append(x)
        if self.use_cat:
            x = torch.cat(hidden_state, dim=1)
        return self.norm(x)


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, d_model, d_ff=1024, num_heads=8, dropout=0.1, local_attention_size=None,
                 make_future=False):
        super(EncoderLayer, self).__init__()
        self.dropout = dropout
        self.d_model = d_model
        self.num_heads = num_heads
        self.local_attention_size = local_attention_size
        self.make_future = make_future
        self.self_attention = MultiHeadedAttention(num_heads, d_model, dropout=self.dropout,
                                                   local_attention_size=None)
        self.feed_forward = PositionwiseFeedForward(d_model=d_model, d_ff=d_ff)

        self.sublayer = clones(ResidualConnection(d_model, dropout), 2)

    def forward(self, x):
        x = self.sublayer[0](x, lambda x: self.self_attention(x, x, x))

        x = self.sublayer[1](x, self.feed_forward)
        return x


class Decoder(nn.Module):
    def __init__(self, input_linear_transform, decode_layer, N):
        super(Decoder, self).__init__()
        self.input_linear_transform = input_linear_transform
        self.layers = clones(decode_layer, N)
        self.norm = nn.BatchNorm1d(decode_layer.d_model)

    def forward(self, x, encoder_output):
        x = self.input_linear_transform(x)
        for decoder_layer in self.layers:
            x = decoder_layer(x, encoder_output)
        return self.norm(x)


class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, d_model, d_ff, num_heads, dropout, local_attention_size):
        super(DecoderLayer, self).__init__()
        self.dropout = dropout
        self.d_model = d_model
        self.num_heads = num_heads
        self.local_attention_size = local_attention_size

        self.self_attention = MultiHeadedAttention(num_heads, d_model, dropout=self.dropout,
                                                   local_attention_size=None)

        self.encoder_attention = MultiHeadedAttention(num_heads, d_model, dropout=self.dropout,
                                                      local_attention_size=None)

        self.feed_forward = PositionwiseFeedForward(d_model=d_model, d_ff=d_ff)
        self.sublayer = clones(ResidualConnection(d_model, dropout), 3)

    def forward(self, x, encoder_output):
        x = self.sublayer[0](x, lambda x: self.self_attention(x, x, x, mask=None))
        x = self.sublayer[1](x, lambda x: self.encoder_attention(x, encoder_output, encoder_output))
        x = self.sublayer[2](x, self.feed_forward)
        return x
