import torch
from torch import nn
import torch.nn.functional as F

from config import Config
from model_utils import *

class Transformer(nn.Module) :
    def __init__(self, args) -> None:
        super().__init__()
        self.InputEmbedding = EmbeddingLayer(args, is_target = False)
        self.TargetEmbedding = EmbeddingLayer(args, is_target=True)
        self.EncoderBlocks = nn.ModuleList([EncoderBlock(args) for num_encoder in range(args.num_blocks)])
        self.DecoderBlocks = nn.ModuleList([DecoderBlock(args) for num_decoder in range(args.num_blocks)])
        self.model_head = DecoderHeadLayer(args)
    
    def forward(self, input_seq, input_pad_idx, target_seq, target_pad_idx, train = True) :
        print(input_pad_idx.device)
        embedded_input, input_pad_idx = self.InputEmbedding(input_seq, input_pad_idx)
        encoder_output = embedded_input
        for encoder_block in self.EncoderBlocks:
            encoder_output = encoder_block(encoder_output, input_pad_idx)
        
        
        embedded_target, target_pad_idx = self.TargetEmbedding(target_seq, target_pad_idx)
        decoder_output = embedded_target

        if train:
            for decoder_block in self.DecoderBlocks:
                decoder_output = decoder_block(decoder_output, target_pad_idx, encoder_output, input_pad_idx)
            
            return self.model_head(decoder_output)
        
        else: # 실제 inference 상황
            pass

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
        self.projection = QKVProjectionSublayer(args)
        self.masked_self_attention = MaskedMultiheadSelfAttentionLayer(args)
        self.residual_block_1 = ResidualConnectionSubLayer(args)
        self.query_projection = nn.Linear(args.embed_dim, args.model_dim)
        self.key_projection = nn.Linear(args.embed_dim, args.model_dim)
        self.value_projection = nn.Linear(args.embed_dim, args.model_dim)
        self.cross_attention = CrossAttentionSubLayer(args)
        self.residual_block_2 = ResidualConnectionSubLayer(args)
        self.linear_transformation = LinearTransformSublayer(args)
        self.residual_block_3 = ResidualConnectionSubLayer(args)

    def forward(self, input_seq, decoder_pad_idxs, encoder_output, encoder_pad_idxs) : # input_seq : (batch, seq_len, embed_dim)
        Q, K, V = self.projection(input_seq)
        masked_self_attention_result = self.masked_self_attention(Q, K, V, decoder_pad_idxs)
        normalized_result_1 = self.residual_block_1(masked_self_attention_result, input_seq)
        Query = self.query_projection(normalized_result_1)
        Key = self.key_projection(encoder_output)
        Value = self.value_projection(encoder_output)
        cross_attention_result = self.cross_attention(Query, Key, Value, decoder_pad_idxs, encoder_pad_idxs)
        normalized_result_2 = self.residual_block_2(cross_attention_result, normalized_result_1)
        linear_transformed_result = self.linear_transformation(normalized_result_2)
        normalized_result_3 = self.residual_block_3(linear_transformed_result, normalized_result_2)
        return normalized_result_3
