class Config():
    def __init__(self) -> None:
        self.vocab_size = 30000
        self.embed_dim = 512
        self.model_dim = 768
        self.pad_idx = 3
        self.max_len = 512
        
        self.sinusoidal = False

        self.ffnn_dim = 4096
