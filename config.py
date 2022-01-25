class Config():
    def __init__(self) -> None:
        self.is_sinusoidal = False
        self.vocab_size = 30000
        self.model_dim = 512
        self.pad_idx = 3
        self.max_len = 512
        
        
        self.sinusoidal = False

        self.ffnn_dim = 2048
