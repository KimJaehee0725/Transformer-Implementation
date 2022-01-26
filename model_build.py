import torch
from torch import nn
from torch.nn import functional as F

from config import Config
from model_utils import *

class Transformer(nn.Module) :
    def __init__(self, args) -> None:
        super().__init__()
        self.Embedding = EmbeddingLayer(args)
        self.EncoderBlocks = [EncoderBlock(args) for num_encoder in range(args.num_blocks)]
        self.DecoderBlocks = [DecoderBlock(args) for num_decoder in range(args.num_blocks)]
        self.SoftmaxHead = nn.Softmax()


class EncoderBlock(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.projection = QKVProjectionSublayer(args)
        self.multihead_attention = MultiHeadSelfAttentionSubLayer(args)
        self.residual_block_1 = ResidualConnectionSubLayer(args)
        self.linear_transformation = LinearTransformSublayer(args)
        self.residual_block_2 = ResidualConnectionSubLayer(args)


    def forward(self, input_seq, pad_idxs): # input_seq : (batch, seq_len , embed_dim) pad_idxs :(batch_size, 1)
        Q, K, V = self.projection(input_seq) # Q, K, V :(batch, seq_len, model_dim)
        attention_result = self.multihead_attention(Q, K, V, pad_idxs) # attention_result : (batch, seq_len, embedding_dim)
        normalized_result_1 = self.residual_block_1(attention_result, input_seq) # normalized_result_1  : (batch, seq_len, model_dim)
        linear_transformed_result = self.linear_transformation(normalized_result_1) # linear_transformed_result :(batch, seq_len, embdding_dim)
        normalized_result_2 = self.residual_block_2(linear_transformed_result, normalized_result_1) # normalized_result_2 : (batch, seq_len, embdding_dim)
        return normalized_result_2

class DecoderBlock(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()