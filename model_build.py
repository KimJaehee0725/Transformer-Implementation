import torch
from torch import nn
import torch.nn.functional as F

from config import Config
from model_utils import *
from model_blocks import EncoderBlock, DecoderBlock


class Transformer(nn.Module) :
    def __init__(self, args) -> None:
        super().__init__()
        self.device = args.device
        self.bos_id = args.bos_id
        self.eos_id = args.eos_id
        self.pad_id = args.pad_id

        self.batch_size = args.batch_size
        self.valid_batch_size = args.valid_batch_size
        self.max_len = args.max_len

        self.InputEmbedding = EmbeddingLayer(args, is_target = False)
        self.TargetEmbedding = EmbeddingLayer(args, is_target=True)
        self.EncoderBlocks = nn.ModuleList([EncoderBlock(args) for num_encoder in range(args.num_blocks)])
        self.DecoderBlocks = nn.ModuleList([DecoderBlock(args) for num_decoder in range(args.num_blocks)])
        self.model_head = DecoderHeadLayer(args)
    
    def forward(self, input_seq, input_pad_idx, target_seq = None, target_pad_idx = None, train = True) :
        embedded_input, input_pad_idx = self.InputEmbedding(input_seq, input_pad_idx)
        encoder_output = embedded_input
        for encoder_block in self.EncoderBlocks:
            encoder_output = encoder_block(encoder_output, input_pad_idx)
        
        if train:
            embedded_target, target_pad_idx = self.TargetEmbedding(target_seq, target_pad_idx)
            decoder_output = embedded_target
            for decoder_block in self.DecoderBlocks:
                decoder_output = decoder_block(decoder_output, target_pad_idx, encoder_output, input_pad_idx)
            
            return self.model_head(decoder_output)
        
        else: # 실제 inference 상황
            decoder_tokens = torch.full(size = (self.valid_batch_size, self.max_len+1), fill_value = self.pad_id, dtype = torch.long, device = self.device)
            decoder_tokens[:, 0] = self.bos_id
            embedded_decoder_input, target_pad_idx = self.TargetEmbedding(decoder_tokens[:, -1:])
            
            for time_step in range(self.max_len):
                decoder_output = embedded_decoder_input
                for decoder_block in self.DecoderBlocks:
                    decoder_output = decoder_block(decoder_output, target_pad_idx, encoder_output, input_pad_idx)
                decoder_output = self.model_head(decoder_output) # (batch_size, max_len, vocab_size)
                decoder_pred = torch.argmax(decoder_output[:, time_step, :], dim = 1) # (batch_size, 1, vocab_size)
                decoder_tokens[:, time_step + 1] = decoder_pred
                if (self.eos_id or self.pad_id) in decoder_tokens[:, time_step + 1]:
                    break

            return decoder_output

