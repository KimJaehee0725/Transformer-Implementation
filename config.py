class Config():
    def __init__(self) -> None:
        self.is_sinusoidal = False
        self.vocab_size = 30000
        self.model_dim = 512
        self.pad_id = 3
        self.max_len = 128
        self.embed_dim = 512
        
        self.embedding_dropout_ratio = 0.1
        self.model_dropout_ratio = 0.1

        self.num_blocks = 6
        
        self.sinusoidal = False

        self.ffnn_dim = 2048
