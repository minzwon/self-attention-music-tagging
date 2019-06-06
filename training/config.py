class Config(object):
    def __init__(self,
                back_end,
                conv_channels,
                attention_channels,
                attention_layers,
                attention_heads,
                attention_length,
                num_class,
                batch_size,
                attention_dropout,
                fc_dropout,
                is_cuda):
        self.back_end = back_end
        self.conv_channels = conv_channels
        self.attention_channels = attention_channels
        self.attention_layers = attention_layers
        self.attention_heads = attention_heads
        self.attention_length = attention_length
        self.num_class = num_class
        self.batch_size = batch_size
        self.attention_dropout = attention_dropout
        self.fc_dropout = fc_dropout
        self.is_cuda = is_cuda
