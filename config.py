class Config():
    def __init__(self) -> None:
        self.is_sinusoidal = False
        self.max_len = 128     

        self.model_dim = 512   
        self.embed_dim = 512
        self.ffnn_dim = 2048

        self.num_blocks = 6

        self.embedding_dropout_ratio = 0.1
        self.model_dropout_ratio = 0.1

        self.sos_token = "<BOS>"
        self.eos_token = "<EOS>"

        self.pad_id = 3

        self.batch_size = 32
        self.epochs = 20

        self.vocab_size = 3000
        self.target_vocab_size = 3000