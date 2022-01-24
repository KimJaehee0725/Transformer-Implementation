from tkinter import Variable
import torch 
from torch import nn
from config import Config
from math import sqrt, sin, cos

args = Config()

class EmbeddingLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.token_embedding = TokenEmbedding(args)
        self.PEembedding = PositionalEmbedding(args)
    
    def forward(self, token_tensor):
        output = self.token_embedding(token_tensor)
        output = self.PEembedding(output)
        return output

class TokenEmbedding(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.vocab_size = args.vocab_size
        self.embedding_dim = args.embedding_dim
        self.pad_idx = args.pad_idx

        self.Embedding_layer = nn.Embedding(
            num_embeddings = self.vocab_size, 
            embedding_dim = self.model_dim,
            padding_idx = self.pad_idx)
        
    def forward(self, token_tensor):
        output = self.Embedding_layer(token_tensor)
        output = output * sqrt(self.embedding_dim)
        return output


class PositionalEmbedding(nn.Module):
    def __init__(self, args):
        super().__init__()
        if args.is_sinusoidal == True:
            self.PEEmbedding = self.makeSinusiodalEmbedding(args)
        else :
            self.PEEmbedding = torch.rand((args.max_len, args.model_dim))
    
    def forward(self, token_embedding):
        return token_embedding + Variable(self.PEEmbedding, requires_grad = True)
    
    def makeSinusiodalEmbedding(self, args):
        embedding_tensor = torch.zeros(args.max_len, args.model_dim)

        even_max = (args.model_dim + 1)//2
        odd_max = args.model_dim//2

        for pos in range(args.max_len):
            pos_even = torch.full(size = (even_max), fill_value = pos)
            pos_even = [sin(elem/10000**(2*num/args.dim_model)) for num, elem in enumerate(pos_even)]
            embedding_tensor[pos, 0::2] = pos_even

            pos_odd = torch.full(size = (odd_max), fill_value = pos)
            pos_odd = [cos(elem/10000**(2*num/args.dim_model)) for num, elem in enumerate(pos_odd)]
            embedding_tensor[pos, 1::2] = pos_odd

        return embedding_tensor
    





class SelfAttentionLayer(nn.Module):
    pass

class FFNNSubLayer(nn.Module):
    pass

class ResidualBlockSubLayer(nn.Module):
    pass

class LMNormailizationSubLayer(nn.Module):
    pass

class CrossAttentionLayer(nn.Module):
    pass

class MaskedAttentionLayer(nn.Module):
    pass

class DecoderHeadLayer(nn.Module):
    pass








class 