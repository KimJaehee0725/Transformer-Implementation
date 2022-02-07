import torch
class Config():
    def __init__(self) -> None:
        self.is_sinusoidal = False
        self.max_len = 32     

        self.model_dim = 512   
        self.embed_dim = 512
        self.ffnn_dim = 2048

        self.num_blocks = 6

        self.embedding_dropout_ratio = 0.1
        self.model_dropout_ratio = 0.1

        self.sos_token = "<BOS>"
        self.eos_token = "<EOS>"

        self.bos_id = 1
        self.eos_id = 2
        self.pad_id = 3
        
        self.batch_size = 64
        self.epochs = 20
        self.valid_epoch = 4
        self.valid_batch_size = 128


        self.vocab_size = 3000
        self.target_vocab_size = 3000

        self.lr = 1e-8
        self.t0 = 250
        self.t_mult = 2
        self.eta_min = 1e-4

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")