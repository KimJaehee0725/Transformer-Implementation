from collections import Counter
from torchtext.vocab import build_vocab_from_iterator
from config import Config
import torch
from torch.utils.data import Dataset

args = Config()

def yield_tokens(corpus_file, tokenizer, is_korean = True):
    for line in corpus_file:
        if is_korean:
            yield tokenizer.morphs(line)
        else:
            yield tokenizer(line)


# def build_vocab(corpus_file, tokenizer, is_korean = False):
#     counter = Counter()
#     for string_ in corpus_file:
#         if is_korean :
#             counter.update(tokenizer.morphs(string_))
#         else:   
#             counter.update(tokenizer(string_))
#     return Vocab(counter, min_freq = 3, specials = ('<unk>', '<BOS>', '<EOS>', "<PAD>"))

class CustomDataset(Dataset):
    def __init__(self, english_file, korean_file, korean_vocab, english_vocab, korean_tokenizer, english_tokenzier, args) -> None:
        super().__init__() 
        self.target = english_file
        self.source = korean_file
        
        self.korean_vocab = korean_vocab
        self.korean_tokenizer = korean_tokenizer

        self.english_vocab = english_vocab
        self.english_tokenizer = english_tokenzier

        self.sos_token = args.sos_token
        self.eos_token = args.eos_token
        assert len(self.target) == len(self.source), "데이터셋 길이가 다릅니다. 확인해보세요."

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index) :
        source_instance, target_instance = self.source[index], self.target[index]
        source_tokens = [self.sos_token] + self.korean_tokenizer.morphs(source_instance) + [self.eos_token]
        target_tokens = [self.sos_token] + self.english_tokenizer(target_instance) + [self.eos_token]

        source_preprocess = self.convert_tokens_to_ids(source_tokens, self.korean_vocab)
        target_preprocess = self.convert_tokens_to_ids(target_tokens, self.english_vocab)
        return (source_preprocess, target_preprocess)
    
    def convert_tokens_to_ids(self, tokens, vocab):
        unk_id = 0
        result = []
        for token in tokens:
            try:
                token_id = vocab[token]
            except:
                token_id = unk_id
            result.append(token_id)
        return result       
    
def collate_fn(batch):
    source_tensor = torch.full(size = (len(batch), args.max_len), fill_value = args.pad_id)
    target_tensor = torch.full(size = (len(batch), args.max_len + 1), fill_value = args.pad_id)
    max_len = args.max_len
    source_pad = []
    target_pad = []

    for num, (source_tokens, target_tokens) in enumerate(batch):
        source_preprocessing = torch.tensor(source_tokens)[:max_len]
        source_len = source_preprocessing.shape[0]
        target_preprocessing = torch.tensor(target_tokens)[:max_len + 1]      
        target_len = target_preprocessing.shape[0]  

        source_tensor[num, :source_len] = source_preprocessing
        target_tensor[num, :target_len] = target_preprocessing

        source_pad.append(source_len + 1)
        target_pad.append(target_len + 1)
    
    return (source_tensor, source_pad), (target_tensor, target_pad)

