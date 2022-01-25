from ast import AsyncFunctionDef
import torch 
from torch import nn
from config import Config
from math import sqrt, sin, cos

args = Config()

class EmbeddingLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.token_embedding = TokenEmbeddingSubLayer(args)
        self.PEembedding = PositionalEmbeddingSubLayer(args)
        self.embedding = nn.Sequential(self.token_embedding, self.PEembedding)
    
    def forward(self, token_tensor): # token_tensor :(batch, seq_len)
        output = self.embedding(token_tensor)
        return output # output : (batch, seq_len, model_dim)

class TokenEmbeddingSubLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.vocab_size = args.vocab_size
        self.embedding_dim = args.model_dim
        self.pad_idx = args.pad_idx

        self.Embedding_layer = nn.Embedding(
            num_embeddings = self.vocab_size, 
            embedding_dim = self.embedding_dim,
            padding_idx = self.pad_idx)
        
    def forward(self, token_tensor):
        output = self.Embedding_layer(token_tensor)
        output = output * sqrt(self.embedding_dim)
        return output


class PositionalEmbeddingSubLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        if args.is_sinusoidal == True:
            self.PEEmbedding = self.makeSinusiodalEmbedding(args)
        else :
            self.PEEmbedding = torch.rand((args.max_len, args.model_dim))
    
    def forward(self, token_embedding):
        return token_embedding + torch.tensor(self.PEEmbedding) #토치 변수 선언(for autograd 이용)
    
    def makeSinusiodalEmbedding(self, args):
        embedding_tensor = torch.zeros(args.max_len, args.model_dim)

        even_max = (args.model_dim + 1)//2
        odd_max = args.model_dim//2

        for pos in range(args.max_len):
            pos_even = [pos]*even_max
            # pos_even = torch.full(size = (1, even_max), fill_value = pos)
            pos_even = torch.tensor([sin(elem/10000**(2*num/args.model_dim)) for num, elem in enumerate(pos_even)])
            embedding_tensor[pos, 0::2] = pos_even

            pos_odd = [pos]*odd_max
            # pos_odd = torch.full(size = (1, odd_max), fill_value = pos)
            pos_odd = torch.tensor([cos(elem/10000**(2*num/args.model_dim)) for num, elem in enumerate(pos_odd)])
            embedding_tensor[pos, 1::2] = pos_odd

        return embedding_tensor
    
class EncoderBlock(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.projection = QKVProjectionSublayer(args)
        self.residual_block_1 = ResidualConnectionSubLayer(args)
        self.linear_transformation = LinearTransformSublayer(args)
        self.residual_block_2 = ResidualConnectionSubLayer(args)


    def forward(self, input_seq): # input_seq : (batch, seq_len , model_dim)
        Q, K, V = self.projection(input_seq) # Q, K, V :(batch, seq_len, model_dim)
        attention_result = self.MultiHeadSelfAttention(Q, K, V) # attention_result : (batch, seq_len, model_dim)
        normalized_result_1 = self.residual_block_1(attention_result, input_seq) # normalized_result_1  : (batch, seq_len, model_dim)
        linear_transformed_result = self.linear_transformation(normalized_result_1) # linear_transformed_result :(batch, seq_len, model_dim)
        normalized_result_2 = self.residual_block_2(linear_transformed_result, normalized_result_1) # normalized_result_2 : (batch, seq_len, model_dim)
        return normalized_result_2

    def MultiHeadSelfAttention(self, Q, K, V): # Q, K, V : (batch_size, seq_len, model_dim)
        model_dim = Q.size()[2]
        K_transposed = K.transpose(1, 2) # K_transposed : (batch_size, model_dim, seq_len)
        attention_matrix = torch.matmul(Q, K_transposed)
        scaled_attention_matrix = attention_matrix/sqrt(model_dim) # scaled_attention_matrix : (batch_size, seq_len, seq_len)
        return torch.matmul(scaled_attention_matrix, V)

class QKVProjectionSublayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.QueryProjection = nn.Linear(args.model_dim, args.model_dim)
        self.KeyProjection = nn.Linear(args.model_dim, args.model_dim)
        self.ValueProjection = nn.Linear(args.model_dim, args.model_dim)
    
    def forward(self, input_seq) : # input_seq : (batch, seq_len, input_dim)
        Q = self.QueryProjection(input_seq) # Q, K, V : (batch, seq_len, model_dim)
        K = self.KeyProjection(input_seq)
        V = self.ValueProjection(input_seq)
        return Q, K, V
        
class ResidualConnectionSubLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.norm_layer = nn.LayerNorm(args.model_dim)

    def forward(self, transformed, original) : # transformed : (batch_size, seq_len, model_dim) original : (batch_size, seq_len, model_dim)
        connected = original + transformed # connected : (batch_size, seq_len, model_dim)
        normalized = self.norm_layer(connected)
        return normalized
    
class LinearTransformSublayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.ffnn_dim = args.ffnn_dim
        self.model_dim = args.model_dim

        self.widen_layer = nn.Linear(self.model_dim, self.ffnn_dim)
        self.activation_function = nn.ReLU()
        self.narrow_layer = nn.Linear(self.ffnn_dim, self.model_dim)
    
    def forward(self, input_seq) : # input_seq :(batch_size, seq_len, model_dim)
        widen_result = self.widen_layer(input_seq) # widen_result : (batch_size, seq_len, ffnn_dim)
        widen_activation_pass = self.activation_function(widen_result)
        return self.narrow_layer(widen_activation_pass) # (batch_size, seq_len, model_dim)


class CrossAttentionLayer(nn.Module):
    pass

class MaskedAttentionLayer(nn.Module):
    pass

class DecoderHeadLayer(nn.Module):
    pass

