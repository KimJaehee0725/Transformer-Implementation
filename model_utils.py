import torch 
from torch import nn
import torch.nn.functional as F 
from config import Config
from math import sqrt, sin, cos

args = Config()

class EmbeddingLayer(nn.Module):
    def __init__(self, args, is_target = False):
        super().__init__()
        self.token_embedding = TokenEmbeddingSubLayer(args, is_target = is_target)
        self.PEembedding = PositionalEmbeddingSubLayer(args)
        self.embedding = nn.Sequential(self.token_embedding, self.PEembedding)
        self.dropout_layer = nn.Dropout(args.embedding_dropout_ratio)
        self.pad_id = args.pad_id

    def forward(self, token_tensor, pad_idxs = None): # token_tensor :(batch, seq_len)
        summed = self.embedding(token_tensor)
        output = self.dropout_layer(summed)

        if pad_idxs == None:
            index_tensor = torch.tensor([[i for i in range(token_tensor.size()[1])] for batch in range(token_tensor.size()[0])])
            masking = (token_tensor != self.pad_id)
            not_pad_mask = index_tensor*masking
            pad_idxs = torch.max(not_pad_mask, dim = 1).values + 1
            

        return output, pad_idxs # output : (batch, seq_len, embed_dim) pad_idxs :(batch_size, 1)

class TokenEmbeddingSubLayer(nn.Module):
    def __init__(self, args, is_target = False):
        super().__init__()
        self.vocab_size = args.target_vocab_size if is_target else args.vocab_size
        self.embedding_dim = args.embed_dim
        self.pad_id = args.pad_id

        self.Embedding_layer = nn.Embedding(
            num_embeddings = self.vocab_size, 
            embedding_dim = self.embedding_dim,
            padding_idx = self.pad_id)
        
    def forward(self, token_tensor):
        output = self.Embedding_layer(token_tensor)
        output = output * sqrt(self.embedding_dim)
        return output

class PositionalEmbeddingSubLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        if args.is_sinusoidal == True:
            PEEmbedding = self.makeSinusiodalEmbedding(args)
        else :
            PEEmbedding = torch.rand((args.max_len, args.embed_dim))
        self.PEEmbedding = nn.Parameter(PEEmbedding)
    
    def forward(self, token_embedding):
        return token_embedding + self.PEEmbedding #토치 변수 선언(for autograd 이용)
    
    def makeSinusiodalEmbedding(self, args):
        embedding_tensor = torch.zeros(args.max_len, args.embed_dim)

        even_max = (args.embed_dim + 1)//2
        odd_max = args.embed_dim//2

        for pos in range(args.max_len):
            pos_even = [pos]*even_max
            # pos_even = torch.full(size = (1, even_max), fill_value = pos)
            pos_even = torch.tensor([sin(elem/10000**(2*num/args.embed_dim)) for num, elem in enumerate(pos_even)])
            embedding_tensor[pos, 0::2] = pos_even

            pos_odd = [pos]*odd_max
            # pos_odd = torch.full(size = (1, odd_max), fill_value = pos)
            pos_odd = torch.tensor([cos(elem/10000**(2*num/args.embed_dim)) for num, elem in enumerate(pos_odd)])
            embedding_tensor[pos, 1::2] = pos_odd

        return embedding_tensor
    
class MultiHeadSelfAttentionSubLayer(nn.Module): # Q, K, V : (batch_size, seq_len, model_dim)
    def __init__(self, args) :
        super().__init__()
        self.ffnn_layer = nn.Linear(args.model_dim, args.embed_dim)

    def forward(self, Q, K, V, pad_idxs):
        model_dim = Q.size()[2]
        K_transposed = K.transpose(1, 2) # K_transposed : (batch_size, model_dim, seq_len)
        attention_score_matrix = torch.matmul(Q, K_transposed)
        attention_score_matrix= attention_score_matrix/sqrt(model_dim) # scaled_attention_score : (batch_size, seq_len, seq_len)
        attention_score_matrix = self.pad_masking(pad_idxs, attention_score_matrix)
        attention_matrix = F.softmax(attention_score_matrix, dim = 2)
        attention_result = torch.matmul(attention_matrix, V)
        return self.ffnn_layer(attention_result)

    def pad_masking(self, pad_idxs, attention_score, not_mask_sign = -1e13):
        for num, pad_start in enumerate(pad_idxs) : # 각 배치마다 수행
            attention_score[num, pad_start:, pad_start:] = not_mask_sign
        return attention_score

class QKVProjectionSublayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.QueryProjection = nn.Linear(args.embed_dim, args.model_dim)
        self.KeyProjection = nn.Linear(args.embed_dim, args.model_dim)
        self.ValueProjection = nn.Linear(args.embed_dim, args.model_dim)
    
    def forward(self, input_seq) : # input_seq : (batch, seq_len, input_dim)
        Q = self.QueryProjection(input_seq) # Q, K, V : (batch, seq_len, model_dim)
        K = self.KeyProjection(input_seq)
        V = self.ValueProjection(input_seq)
        return Q, K, V
        
class ResidualConnectionSubLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.norm_layer = nn.LayerNorm(args.model_dim)
        self.dropout_layer = nn.Dropout(args.model_dropout_ratio)

    def forward(self, transformed, original) : # transformed : (batch_size, seq_len, embed_dim) original : (batch_size, seq_len, embed_dim)
        dropped = self.dropout_layer(transformed)
        connected = original + dropped # connected : (batch_size, seq_len, embed_dim)
        normalized = self.norm_layer(connected)
        return normalized
    
class LinearTransformSublayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.ffnn_dim = args.ffnn_dim
        self.model_dim = args.model_dim
        self.embdding_dim = args.embed_dim

        self.widen_layer = nn.Linear(self.model_dim, self.ffnn_dim)
        self.activation_function = nn.ReLU()
        self.narrow_layer = nn.Linear(self.ffnn_dim, self.embdding_dim)
    
    def forward(self, input_seq) : # input_seq :(batch_size, seq_len, model_dim)
        widen_result = self.widen_layer(input_seq) # widen_result : (batch_size, seq_len, ffnn_dim)
        widen_activation_pass = self.activation_function(widen_result)
        return self.narrow_layer(widen_activation_pass) # (batch_size, seq_len, embdding_dim)


class CrossAttentionSubLayer(nn.Module):
    def __init__(self, args) :
        super().__init__()
        self.ffnn_layer = nn.Linear(args.model_dim, args.embed_dim)

    def forward(self, Q, K, V, query_pad_idxs, key_pad_idxs):
        model_dim = Q.size()[2]
        K_transposed = K.transpose(1, 2) # K_transposed : (batch_size, model_dim, seq_len)
        attention_score_matrix = torch.matmul(Q, K_transposed)
        attention_score_matrix= attention_score_matrix/sqrt(model_dim) # scaled_attention_score : (batch_size, seq_len, seq_len)
        attention_score_matrix = self.pad_masking(query_pad_idxs, key_pad_idxs, attention_score_matrix)
        attention_matrix = F.softmax(attention_score_matrix, dim = 2)
        attention_result = torch.matmul(attention_matrix, V)
        return self.ffnn_layer(attention_result)

    def pad_masking(self, query_pad_idxs, key_pad_idxs, attention_score, not_mask_sign = -1e13):
        for num, (query_pad, key_pad) in enumerate(zip(query_pad_idxs, key_pad_idxs)) : # 각 배치마다 수행
            attention_score[num, query_pad:, key_pad:] = not_mask_sign
        return attention_score

class MaskedMultiheadSelfAttentionLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.ffnn_layer = nn.Linear(args.model_dim, args.embed_dim)

    def forward(self, Q, K, V, pad_idxs):
        model_dim = Q.size()[2] # Q, K, V : (batch_size, seq_len, embed_dim)
        K_transposed = K.transpose(1, 2) # K_transposed : (batch_size, model_dim, seq_len)
        attention_score_matrix = torch.matmul(Q, K_transposed)
        attention_score_matrix= attention_score_matrix/sqrt(model_dim) # scaled_attention_score : (batch_size, seq_len, seq_len)
        attention_score_matrix = self.pad_masking(pad_idxs, attention_score_matrix)
        attention_score_matrix = self.self_masking(attention_score_matrix)
        attention_matrix = F.softmax(attention_score_matrix, dim = 2)
        attention_result = torch.matmul(attention_matrix, V)
        return self.ffnn_layer(attention_result)

    def pad_masking(self, pad_idxs, attention_score, not_attention_sign = -1e13):
        for num, pad_start in enumerate(pad_idxs) : # 각 배치마다 수행
            attention_score[num, pad_start:, pad_start:] = not_attention_sign
        return attention_score
    
    def self_masking(self, attention_score, not_attention_sign = -1e13):
        for num in range(attention_score.size()[1]) : # 배치 동시 수행
            attention_score[:, num, (num+1):] = not_attention_sign
        return attention_score

class DecoderHeadLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.linear = nn.Linear(args.embed_dim, args.target_vocab_size)
    
    def forward(self, input_seq) : # input_seq : (batch_size, seq_len, embed_dim)
        transform = self.linear(input_seq)
        return F.log_softmax(transform, dim = 1)


