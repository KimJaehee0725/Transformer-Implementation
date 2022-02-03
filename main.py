from copy import deepcopy

from konlpy.tag import Mecab

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

def main():
    torch.cuda.empty_cache() 
    args = Config()

    wandb.init(config = args, project = "Transofrmer_Implementation", name = "The First Trasformer Implementation")

    korean_tokenizer = Mecab()
    english_tokenizer = get_tokenizer('moses')

    with open(f"../data/train/korean-english-park.train.en", encoding = "utf-8")as f:
        train_eng = f.readlines()
        train_eng = [sentence.replace("\n", "") for sentence in train_eng]
    with open(f"../data/train/korean-english-park.train.ko", encoding = "utf-8")as f:
        train_ko = f.readlines()
        train_ko = [sentence.replace("\n", "") for sentence in train_eng]
    with open(f"../data/dev/korean-english-park.dev.en", encoding = "utf-8")as f:
        valid_eng = f.readlines()
        valid_eng = [sentence.replace("\n", "") for sentence in train_eng]
    with open(f"../data/dev/korean-english-park.dev.ko", encoding = "utf-8")as f:
        valid_ko = f.readlines()
        valid_ko = [sentence.replace("\n", "") for sentence in train_eng]
        
    korean_vocab = build_vocab_from_iterator(yield_tokens(train_ko, korean_tokenizer, is_korean = True), min_freq = 5, specials = ('<unk>', '<BOS>', '<EOS>', "<PAD>"))
    english_vocab = build_vocab_from_iterator(yield_tokens(train_eng, english_tokenizer, is_korean=False), min_freq = 5, specials = ('<unk>', '<BOS>', '<EOS>', "<PAD>"))

    train_dataset = CustomDataset(train_eng, train_ko, korean_vocab, english_vocab, korean_tokenizer, english_tokenizer, args)
    train_dataloader = DataLoader(train_dataset, batch_size = args.batch_size, collate_fn = collate_fn, shuffle = True, drop_last = True)

    valid_dataset = CustomDataset(valid_eng, valid_ko, korean_vocab, english_vocab, korean_tokenizer, english_tokenizer, args)
    valid_dataloader = DataLoader(valid_dataset, batch_size = args.valid_batch_size, collate_fn = collate_fn)

    args.vocab_size = len(korean_vocab)
    args.target_vocab_size = len(english_vocab)

    model = Transformer(args)

    def train(model, train_dataloader, validation_dataloader, args):    
        device = torch.device("cuda")
        model.to(device)
        model.train()
        optimizer = Adam(model.parameters(), lr = args.lr)
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer = optimizer, T_0 = args.t0, T_mult = args.t_mult, eta_min = args.eta_min)
        criterion = nn.NLLLoss(ignore_index = args.pad_id)
        len_train_batch = len(train_dataloader)
        for epoch in range(args.epochs):

            if epoch%args.valid_epoch == 0 :
                valid_loss = validation(model, validation_dataloader, args)
                wandb.log({"valid_loss" : valid_loss, "epoch" : epoch})
        
            print("training step 시작")
            for num, ((source_tensor, source_pad), (target_tensor, target_pad)) in enumerate(train_dataloader):

                if num%100 == 0 :
                    print(f"{num | len_train_batch | round((num/len_train_batch), 2)*100}%", end = "\r")
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

                lr = scheduler.get_last_lr()[0]

                wandb.log({"loss" : loss, 'epoch' : epoch, "lr" : lr})

                loss.backward()

                optimizer.step()
                scheduler.step()

    def validation(model, dataloader, args):
        print("validation step 시작")
        device = torch.device("cuda")
        model.to(device)
        model.eval()
        criterion = nn.NLLLoss(ignore_index = args.pad_id)
        loss_total = 0
        batch_len = len(dataloader)

        with torch.no_grad():
            for num, ((source_tensor, source_pad), (target_tensor, target_pad)) in enumerate(dataloader):
                if num%30 == 0 :
                    print(f"{round(num/batch_len * 100, 2)}% 완료")

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
        print("validation step 종료")
        return loss_total/batch_len

    train(model, train_dataloader, valid_dataloader, args)

if __name__ == "__main__" :
    main()