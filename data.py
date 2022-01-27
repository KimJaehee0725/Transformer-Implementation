from collections import Counter
from torchtext import Vocab
def build_vocab(train_path, tokenizer):
    counter = Counter()
    with open(train_path, encoding = 'UTF-8', newline = '\n') as f:
        for string_ in f:
            if 'ko' in train_path[-10:]:
                counter.update(tokenizer.morphs(string_))
            else:   
                counter.update(tokenizer(string_))
        return Vocab(counter, min_freq = 3, specials = ('<unk>', '<BOS>', '<EOS>', "<PAD>"))