from copy import deepcopy

from konlpy.tag import Mecab
from numpy import source

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext.data.utils import get_tokenizer
from torch.utils.data import DataLoader
from torchtext.vocab import build_vocab_from_iterator
from torch.optim import lr_scheduler, Adam

from model_build import Transformer
from config import Config
from data import CustomDataset, collate_fn, yield_tokens

import wandb

args = Config()

wandb.init(config = args, project = "Transofrmer_Implementation", )

korean_tokenizer = Mecab()
english_tokenizer = get_tokenizer('moses')

with open(f"../data/korean-english-news-v1/train/korean-english-park.train.en", encoding = "utf-8")as f:
    train_eng = f.readlines()
    train_eng = [sentence.replace("\n", "") for sentence in train_eng]
with open(f"../data/korean-english-news-v1/train/korean-english-park.train.ko", encoding = "utf-8")as f:
    train_ko = f.readlines()
    train_ko = [sentence.replace("\n", "") for sentence in train_eng]
with open(f"../data/korean-english-news-v1/valid/korean-english-park.train.en", encoding = "utf-8")as f:
    valid_eng = f.readlines()
    valid_eng = [sentence.replace("\n", "") for sentence in train_eng]
with open(f"../data/korean-english-news-v1/valid/korean-english-park.train.ko", encoding = "utf-8")as f:
    valid_ko = f.readlines()
    valid_ko = [sentence.replace("\n", "") for sentence in train_eng]
    
korean_vocab = build_vocab_from_iterator(yield_tokens(train_ko, korean_tokenizer, is_korean = True), min_freq = 5, specials = ('<unk>', '<BOS>', '<EOS>', "<PAD>"))
english_vocab = build_vocab_from_iterator(yield_tokens(train_eng, english_tokenizer, is_korean=False), min_freq = 5, specials = ('<unk>', '<BOS>', '<EOS>', "<PAD>"))

train_dataset = CustomDataset(train_eng, train_ko, korean_vocab, english_vocab, korean_tokenizer, english_tokenizer, args)
train_dataloader = DataLoader(train_dataset, batch_size = args.batch_size, collate_fn = collate_fn)

valid_dataset = CustomDataset(valid_eng, valid_ko, korean_vocab, english_vocab, korean_tokenizer, english_tokenizer, args)
valid_dataloader = DataLoader(valid_dataset, batch_size = args.batch_size, collate_fn = collate_fn)

args.vocab_size = len(korean_vocab)
args.target_vocab_size = len(english_vocab)

model = Transformer(args)

def train(model, train_dataloader, validation_dataloader, args):    
    device = torch.device("cuda")
    model.to(device)
    model.train()
    optimizer = Adam(model.parameters(), lr = 1e-3)
    scheduler = lr_scheduler.CosineAnnealingLRWarmRestarts(optimizer = optimizer, T_0 = args.epochs, T_mult = 2, eta_min = 1e-5)

    criterion = nn.NLLLoss(ignore_index = args.pad_id)
    for epoch in range(args.epochs):

        if epoch//args.valid_epoch == 0 :
            valid_loss = validation(model, validation_dataloader, args)
            wandb.log({"valid_loss" : valid_loss, "epoch" : epoch})

        for (source_tensor, source_pad), (target_tensor, target_pad) in train_dataloader:
            optimizer.zero_grad()

            source_tensor = source_tensor.to(device)
            source_pad_tensor = torch.tensor(source_pad, device = device)
            
            target_input = deepcopy(target_tensor[:, :-1])
            target_output = deepcopy(target_tensor[:, 1:])
            
            target_input = target_input.to(device)
            target_output = target_output.to(device)
            target_pad_tensor = torch.tensor(target_pad, device = device)

            output = model(source_tensor, source_pad_tensor, target_input, target_pad_tensor)

            loss = criterion(output.transpose(1, 2), target_output)

            wandb.log({"loss" : loss, 'epoch' : epoch})

            loss.backward()

            optimizer.step()
            scheduler.step()

def validation(model, dataloader, args):
    device = torch.device("cuda")
    model.to(device)
    model.eval()
    criterion = nn.NLLLoss(ignore_index = args.pad_id)

    loss_total = 0
    with torch.no_grad():
        for (source_tensor, source_pad), (target_tensor, target_pad) in dataloader:
            source_tensor = source_tensor.to(device)
            source_pad_tensor = torch.tensor(source_pad, device = device)
            
            target_input = deepcopy(target_tensor[:, :-1])
            target_output = deepcopy(target_tensor[:, 1:])
            
            target_input = target_input.to(device)
            target_output = target_output.to(device)
            target_pad_tensor = torch.tensor(target_pad, device = device)

            output = model(source_tensor, source_pad_tensor, target_input, target_pad_tensor)

            loss = criterion(output.transpose(1, 2), target_output).detach().cpu().numpy()

            loss_total += loss.sum()
    return loss_total

train(model, train_dataloader, valid_dataloader, args)