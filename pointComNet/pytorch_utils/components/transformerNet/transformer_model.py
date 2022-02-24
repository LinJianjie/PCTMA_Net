import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from components.transformerNet.attention_module import *
from components.transformerNet.transformerEncoder_Decoder import *
from components.transformerNet.transformer_utils import *
from components.netUtils import NetUtil, CPointNetLinear


# For us no embedding layer is required
# def transformer_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, num_head=8, dropout=0.1):
#     "Helper: Construct a model from hyperparameters."
#     attn = MultiHeadedAttention(num_head, d_model)
#     ff = PositionwiseFeedForward(d_model, d_ff, dropout)
#     position = PositionalEncoding(d_model, dropout)
#     model = EncoderDecoder(Encoder(EncoderLayer(d_model, copy.deepcopy(attn), copy.deepcopy(ff), dropout), N),
#                            Decoder(DecoderLayer(d_model, copy.deepcopy(attn), copy.deepcopy(attn), copy.deepcopy(ff),
#                                                 dropout), N),
#                            nn.Sequential(Embeddings(d_model, src_vocab), copy.deepcopy(position)),
#                            nn.Sequential(Embeddings(d_model, tgt_vocab), copy.deepcopy(position)),
#                            transformer_last_layer(d_model, tgt_vocab))
#
#     # This was important from their code.
#     # Initialize parameters with Glorot / fan_avg.
#     for p in model.parameters():
#         if p.dim() > 1:
#             nn.init.xavier_uniform(p)
#     return model
class NetTEncoder(nn.Module):
    def __init__(self, src_vocab, local_attention_size, N=6, d_model=512, d_ff=1024, num_head=8, dropout=0.1,
                 use_cmlp=False, channels=None):
        super(NetTEncoder, self).__init__()
        self.num_Encoder_Decoder = N
        self.encoder = Encoder(
            input_linear_transform=nn.Sequential(Embeddings(vocab=src_vocab, d_model=d_model, use_cmlp=use_cmlp)),
            encode_layer=EncoderLayer(d_model=d_model, d_ff=d_ff,
                                      num_heads=num_head, dropout=dropout,
                                      local_attention_size=None),
            N=self.num_Encoder_Decoder)
        # channels.insert(0, d_model * self.num_Encoder_Decoder)
        # self.pn_layers = NetUtil.SeqPointNetConv1d(channels)
        self.last_layer_size = 1024
        self.pn_layers = NetUtil.SeqLinear(
            channels=[self.encoder.last_layer_size, self.last_layer_size, self.last_layer_size],
            activation="LeakyReLU")

    def forward(self, src_):
        B, N, D = src_.shape
        z_hat = self.encoder(src_)
        z_hat = self.pn_layers(z_hat)
        # z_hat = torch.max(z_hat, 2, keepdim=True)[0]
        # z_hat = z_hat.view(B, -1)

        return z_hat


class Transformer_Model(nn.Module):
    def __init__(self, src_vocab, local_attention_size, N=6, d_model=512, d_ff=1024, num_head=8, dropout=0.1,
                 max_len=5000, use_cmlp=False, channels=None):
        super(Transformer_Model, self).__init__()
        self.src_vocab = src_vocab
        self.num_Encoder_Decoder = N
        self.d_model = d_model
        self.num_head = num_head
        self.dropout = dropout
        self.max_len = max_len
        self.d_ff = d_ff
        self.local_attention_size = local_attention_size
        self.Encoder = Encoder(
            input_linear_transform=nn.Sequential(Embeddings(self.src_vocab, d_model, use_cmlp=use_cmlp)),
            encode_layer=EncoderLayer(d_model=self.d_model, d_ff=self.d_ff, num_heads=self.num_head,
                                      dropout=self.dropout,
                                      local_attention_size=None),
            N=self.num_Encoder_Decoder,
            use_cat=False)
        self.Decoder = Decoder(
            input_linear_transform=nn.Sequential(Embeddings(self.src_vocab, d_model, use_cmlp=use_cmlp)),
            decode_layer=DecoderLayer(d_model=self.d_model, d_ff=self.d_ff, num_heads=self.num_head,
                                      dropout=self.dropout,
                                      local_attention_size=None),
            N=self.num_Encoder_Decoder)
        self.model = TransformerEncoderDecoder(self.Encoder,
                                               self.Decoder,
                                               transformer_last_layer(d_model, 3))
        # for p in self.model.parameters():
        #      if p.dim() > 1:
        #          nn.init.xavier_uniform(p)

    def forward(self, src_input, target):
        output_encode, output_decode = self.model(src_input, target)
        return output_encode, output_decode

    def forward_evaluate_encoder(self, src_input):
        out_encode = self.model.forward_encode(src_input)
        return out_encode

    def forward_evaluate_decoder(self, out_encode, predicted):
        out_decode = self.model.forward_decode(output_encode=out_encode, target=predicted)
        return out_decode


# def loss_G(x, x_hat, z, z_hat, f1_fake, f1_real, alpha1, alpha2, alpha3):
#     consistent_loss_z = torch.mean(torch.norm(z - z_hat))
#     reconstruction_loss_x = torch.mean(torch.norm(x - x_hat))
#     adversarial_loss = torch.mean(torch.norm(f1_fake - f1_real))
#     return alpha1 * consistent_loss_z + alpha2 * reconstruction_loss_x + alpha3 * adversarial_loss
#
#
# def loss_D(f2_fake, f2_real):
#     return -f2_fake + f2_real
if __name__ == '__main__':
    # TODO test
    # src = torch.rand(8, 200, 20)
    # tgt = torch.rand(8, 100, 20)
    # # transformderModel = Transformer_Model
    # print("check Embeddings")
    # emd = Embeddings(20, 200)
    # linar_src = emd(src)
    # print(linar_src.shape)
    # linear_tgt = emd(tgt)
    # print(linear_tgt.shape)
    # print("check PositionalEncoding")
    # pe = PositionalEncoding(30, dropout=0.1)
    # # encoderlayer = EncoderLayer(d_model=200, num_head=6, dropout=0.1, local_attention_size=20)
    # src_t = torch.rand(8, 9, 40)
    # print("check attention")
    # y, m = attention(src_t, src_t, src_t, mask="subsequent", dropout=0.1)  # checked
    #
    # print("check MultiHeadedAttention")
    # multihead = MultiHeadedAttention(d_model=40, num_heads=4, dropout=0.1, local_attention_size=3)
    # res = multihead(src_t, src_t, src_t, mask=None)
    # print("res: ", res.shape)
    # print("check the EncoderLayer")
    # encoderlayer = EncoderLayer(d_model=40, d_ff=1024, num_heads=4, dropout=0.1, local_attention_size=3)
    # encoderlayer_Res = encoderlayer(src_t)
    # print("encoderlayer_Res: ", encoderlayer_Res.shape)
    # print("check the Encoder")
    # encoderfunc = Encoder(EncoderLayer(d_model=40, d_ff=1024, num_heads=4, dropout=0.1, local_attention_size=3), 4)
    # encoder_res = encoderfunc(src_t)
    # print("encoder_res: ", encoder_res.shape)
    # tgt_t = torch.rand(8, 20, 40)
    #
    # print("check DecoderLayer")
    # decoderlayer = DecoderLayer(d_model=40, d_ff=1024, num_heads=4, dropout=0.1, local_attention_size=3)
    # decoderlayer_res = decoderlayer(x=tgt_t, encoder_output=encoder_res)
    # print("decoderLayer_res: ", decoderlayer_res.shape)
    #
    # print("check Decoder")
    # Decoderfunc = Decoder(DecoderLayer(d_model=40, d_ff=1024, num_heads=4, dropout=0.1, local_attention_size=3), 4)
    # decoder_res = Decoderfunc(x=tgt_t, encoder_output=encoder_res)
    # print("decoder_res: ", decoder_res.shape)

    print("-----------------------------------")
    src = torch.rand(8, 200, 20)
    tgt = torch.rand(8, 200, 20)
    z = torch.rand(8, 200, 512)
    # transformer_model = Transformer_Model(src_vocab=20, tgt_vocab=20, local_attention_size=3, num_head=4)
    # y = transformer_model(src, tgt)
    # netD_ = NetD(src_vocab=20, tgt_vocab=20, local_attention_size=None, num_head=4)
    # x, y = netD_(src, z)
