# coding: utf-8
import numpy as np
import torch
import torch.nn as nn
from .bert_attention import BertConfig, BertEncoder, BertEmbeddings, BertPooler, PositionalEncoding


class AttentionModule(nn.Module):
    def __init__(self, config):
        super(AttentionModule, self).__init__()
        # Configuration
        bert_config = BertConfig(vocab_size=config.attention_channels,
                            hidden_size=config.attention_channels,
                            num_hidden_layers=config.attention_layers,
                            num_attention_heads=config.attention_heads,
                            intermediate_size=config.attention_channels*4,
                            hidden_act="gelu",
                            hidden_dropout_prob=config.attention_dropout,
                            max_position_embeddings=config.attention_length,
                            attention_probs_dropout_prob=config.attention_dropout)

        # Embedding (Feature map + Positional encoding + Mask)
        #self.embedding = BertEmbeddings(bert_config)
        self.embedding = PositionalEncoding(bert_config)

        # Bert encoder
        self.encoder = BertEncoder(bert_config)

        # Bert pooler
        self.pooler = BertPooler(bert_config)

    def forward(self, x):
        x = self.embedding(x)
        encoded_layers = self.encoder(x)
        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output)
        return pooled_output
