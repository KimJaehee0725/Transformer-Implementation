from copy import deepcopy

from konlpy.tag import Mecab
from numpy import source

import torch
import torch.nn.functional as F
from torchtext.data.utils import get_tokenizer
from torch.utils.data import DataLoader
from torchtext.vocab import build_vocab_from_iterator
from torch.optim import lr_scheduler, Adam
from transformers.optimization import get_cosine_schedule_with_warmup

from model_build import Transformer
from config import Config
from data import CustomDataset, collate_fn, yield_tokens


args = Config()
korean_tokenizer = Mecab()
english_tokenizer = get_tokenizer('moses')

with open(f"../data/korean-english-news-v1/train/korean-english-park.train.en", encoding = "utf-8")as f:
    train_eng = f.readlines()
    train_eng = [sentence.replace("\n", "") for sentence in train_eng]
with open(f"../data/korean-english-news-v1/train/korean-english-park.train.ko", encoding = "utf-8")as f:
    train_ko = f.readlines()
    train_ko = [sentence.replace("\n", "") for sentence in train_eng]
    
korean_vocab = build_vocab_from_iterator(yield_tokens(train_ko, korean_tokenizer, is_korean = True), min_freq = 5, specials = ('<unk>', '<BOS>', '<EOS>', "<PAD>"))
english_vocab = build_vocab_from_iterator(yield_tokens(train_eng, english_tokenizer, is_korean=False), min_freq = 5, specials = ('<unk>', '<BOS>', '<EOS>', "<PAD>"))

# korean_vocab = build_vocab(train_ko, korean_tokenizer, is_korean = True)
# english_vocab = build_vocab(train_eng, english_tokenizer, is_korean = False)

dataset = CustomDataset(train_eng, train_ko, korean_vocab, english_vocab, korean_tokenizer, english_tokenizer, args)
dataloader = DataLoader(dataset, batch_size = args.batch_size, collate_fn = collate_fn)

args.vocab_size = len(korean_vocab)
args.target_vocab_size = len(english_vocab)

model = Transformer(args)
device = torch.device("cuda")
model = model.to(device)
# optimizer = Adam(model.parameters(), lr = 1e-3)

scheduler = lr_scheduler.CosineAnnealingLR(optimizer = optimizer, T_max = 50, eta_min = 0)

def train(model, dataloader, args, optimizer, scheduler):    
    for (source_tensor, source_pad), (target_tensor, target_pad) in dataloader:
        device = torch.device("cuda")

        source_tensor = source_tensor.to(device)
        source_pad_tensor = torch.tensor(source_pad, device = device)
        
        target_input = deepcopy(target_tensor[:, :-1])
        target_output = deepcopy(target_tensor[:, 1:])
        
        target_input = target_input.to(device)
        target_output = target_output.to(device)
        target_pad_tensor = torch.tensor(target_pad, device = device)

        output = model(source_tensor, source_pad_tensor, target_input, target_pad_tensor)
        print("x")

        loss = F.nll_loss(output, target_output)
        print(loss)
        break

train(model, dataloader, args, optimizer, scheduler)