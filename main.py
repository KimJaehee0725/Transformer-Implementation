from copy import deepcopy

from konlpy.tag import Mecab

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext.data.utils import get_tokenizer
from torch.utils.data import DataLoader
from torchtext.vocab import build_vocab_from_iterator
from torch.optim import Adam
from torch.cuda.amp import autocast

from model_build import Transformer
from config import Config
from data import CustomDataset, collate_fn, yield_tokens
from schedulers import CosineAnnealingWarmUpRestarts

import wandb

def main():
    
    torch.cuda.empty_cache() 
    args = Config()

    wandb.init(config = args, project = "Transofrmer_Implementation", name = "Trasformer Implementation-CustomSchelduler")
    # wandb.init(config = args, project = "Debugging", name = "test time inference")

    korean_tokenizer = Mecab()
    english_tokenizer = get_tokenizer('moses')

    dataset_dir = "../../datasets/korean-parallel-corpora/korean-english-news-v1"

    with open(f"{dataset_dir}/train/korean-english-park.train.en", encoding = "utf-8")as f:
        train_eng = f.readlines()
        train_eng = [sentence.replace("\n", "") for sentence in train_eng]
    with open(f"{dataset_dir}/train/korean-english-park.train.ko", encoding = "utf-8")as f:
        train_ko = f.readlines()
        train_ko = [sentence.replace("\n", "") for sentence in train_eng]
    with open(f"{dataset_dir}/dev/korean-english-park.dev.en", encoding = "utf-8")as f:
        valid_eng = f.readlines()
        valid_eng = [sentence.replace("\n", "") for sentence in train_eng]
    with open(f"{dataset_dir}/dev/korean-english-park.dev.ko", encoding = "utf-8")as f:
        valid_ko = f.readlines()
        valid_ko = [sentence.replace("\n", "") for sentence in train_eng]
        
    korean_vocab = build_vocab_from_iterator(yield_tokens(train_ko, korean_tokenizer, is_korean = True), min_freq = 5, specials = ('<unk>', '<BOS>', '<EOS>', "<PAD>"))
    english_vocab = build_vocab_from_iterator(yield_tokens(train_eng, english_tokenizer, is_korean=False), min_freq = 5, specials = ('<unk>', '<BOS>', '<EOS>', "<PAD>"))

    train_dataset = CustomDataset(train_eng, train_ko, korean_vocab, english_vocab, korean_tokenizer, english_tokenizer, args)
    train_dataloader = DataLoader(train_dataset, batch_size = args.batch_size, collate_fn = collate_fn, shuffle = True, drop_last = True)

    valid_dataset = CustomDataset(valid_eng[:1000], valid_ko[:1000], korean_vocab, english_vocab, korean_tokenizer, english_tokenizer, args)
    valid_dataloader = DataLoader(valid_dataset, batch_size = args.valid_batch_size, collate_fn = collate_fn)

    args.vocab_size = len(korean_vocab)
    args.target_vocab_size = len(english_vocab)

    model = Transformer(args)

    def train(model, train_dataloader, validation_dataloader, args):    
        device = torch.device("cuda")
        model.to(device)
        model.train()
        optimizer = Adam(model.parameters(), lr = args.lr)
        scheduler = CosineAnnealingWarmUpRestarts(optimizer = optimizer, T_0 = args.t0, T_mult = args.t_mult, eta_max = args.eta_max, T_up = args.T_up, gamma = args.gamma)
        criterion = nn.NLLLoss(ignore_index = args.pad_id)
        len_train_batch = len(train_dataloader)

        scaler = torch.cuda.amp.GradScaler()
        for epoch in range(args.epochs):

            if (epoch+1)%args.valid_epoch == 0 :
                valid_loss = validation(model, validation_dataloader, epoch, args)
                wandb.log({"valid_loss" : valid_loss, "epoch" : epoch})

            print(f"training step 시작 | {epoch + 1} | {round((epoch +1)/args.epochs * 100, 2)}%")
            for num, ((source_tensor, source_pad), (target_tensor, target_pad)) in enumerate(train_dataloader):

                if num%100 == 0 :
                    print(f"{num} | {len_train_batch} | {round((num/len_train_batch)*100, 2)}%", end = "\r")
                optimizer.zero_grad()

                source_tensor = source_tensor.to(device)
                source_pad_tensor = torch.tensor(source_pad, device = device)
                
                target_input = deepcopy(target_tensor[:, :-1])
                target_output = deepcopy(target_tensor[:, 1:])
                
                target_input = target_input.to(device)
                target_output = target_output.to(device)
                target_pad_tensor = torch.tensor(target_pad, device = device)

                with autocast():
                    output = model(source_tensor, source_pad_tensor, target_input, target_pad_tensor)
                    loss = criterion(output.transpose(1, 2), target_output)

                    lr = scheduler.get_lr()[0]

                    wandb.log({"loss" : loss, 'epoch' : epoch, "lr" : lr})
                    
                    scaler.scale(loss).backward()
                    # loss.backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    # optimizer.step()
                    scheduler.step()

    def validation(model, dataloader, epoch, args):
        print("validation step 시작")
        device = torch.device("cuda")
        model.to(device)
        model.eval()
        criterion = nn.NLLLoss(ignore_index = args.pad_id)
        loss_total = 0
        batch_len = len(dataloader)

        log_num = 0

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

                output = model(source_tensor, source_pad_tensor, target_input, target_pad_tensor, train = False)

                loss = criterion(output.transpose(1, 2), target_output).detach().cpu().numpy()

                loss_total += loss.sum()

                if num == 0 :
                    generated = []
                    for samples in output[:10, :, :]:
                        generated.append(torch.argmax(samples, dim = 1).cpu().tolist())
                    generated = [[token for token in sentence if token != args.pad_id] for sentence in generated ]
                    generated = [english_vocab.lookup_tokens(sentence) for sentence in generated]
                    generated = [" ".join(sentence) for sentence in generated]

                    label = [[token for token in sentence if token != args.pad_id] for sentence in target_output[:10, :].cpu().tolist()]
                    label = [" ".join(english_vocab.lookup_tokens(sentence)) for sentence in  label]

                    text_table = wandb.Table(columns = ["epoch", "loss", "generated", "real"])
                    for gener, lab in zip(generated, label) : 
                        text_table.add_data(epoch, loss, gener, lab)
                    wandb.log({f"valid_samples_{log_num}" : text_table})
                    log_num+=1
        print("validation step 종료")
        model.train()
        return loss_total/batch_len

    train(model, train_dataloader, valid_dataloader, args)

if __name__ == "__main__" :
    main()