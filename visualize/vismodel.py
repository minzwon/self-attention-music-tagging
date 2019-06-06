import os
import math
import copy

import numpy as np
import librosa.display
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

# Load re-usable codes
import sys
sys.path.append('../training')
from config import Config
from front_end.pons_module import PonsConvModule
from back_end.bert_attention import BertConfig, BertIntermediate, BertOutput, BertSelfOutput, BertEmbeddings, BertPooler


def get_tensor(fn, num_chunks=1, audio_path=None, input_length=256, dump_spec_to=None, cmap='coolwarm'):
    # load audio
    spec_path = os.path.join(audio_path, 'spec', fn.split('/')[1][:-3]) + 'npy'
    spec = np.load(spec_path, mmap_mode='r')
    # split chunk
    length = spec.shape[1]
    chunk_length = input_length
    hop = (length - chunk_length) // num_chunks
    x = torch.zeros(num_chunks, 96, chunk_length)
    for i in range(num_chunks):
        x[i] = torch.Tensor(spec[:, i*hop:i*hop+chunk_length]).unsqueeze(0)
        if dump_spec_to:
            plt.figure(figsize=(12, 4))
            # camp: coolwarm, magma
            librosa.display.specshow(spec[:, i*hop:i*hop+chunk_length], cmap=cmap)
            fname = dump_spec_to if dump_spec_to.endswith('png') else os.path.join(dump_spec_to, '{}.png'.format(fn.split('/')[1][:-4]))
            plt.savefig(fname, bbox_inches='tight', pad_inches=0)
            plt.close()
    return x


def save_heatmap_fig(data, fname, figsize=(12, 1)):
    plt.figure(figsize=(12, 1))
    data = np.tile(data, (1, 1))
    assert data.shape == (1, 257), data.shape
    sns.heatmap(data, cmap="Reds", xticklabels=False, yticklabels=False, cbar=False)
    plt.savefig(fname, bbox_inches='tight', pad_inches=0)
    plt.close()


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, manual_attention_probs=None,
                manual_attention_idx=None, return_probs=False, fig_base_name=None):
        if manual_attention_probs is not None:
            mixed_value_layer = self.value(hidden_states)
            value_layer = self.transpose_for_scores(mixed_value_layer)
            attention_probs = nn.Softmax(dim=-1)(manual_attention_probs)
            attention_probs = self.dropout(attention_probs)

            context_layer = torch.matmul(attention_probs, value_layer)
            context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
            new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
            context_layer = context_layer.view(*new_context_layer_shape)
            return context_layer

        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if manual_attention_idx is not None:
            # replace attention probs to manual attention probs
            attention_probs = torch.zeros(attention_scores.size())
            attention_probs[:, :, :, manual_attention_idx] = 1
            attention_probs = attention_probs.cuda()
        else:
            # Normalize the attention scores to probabilities.
            attention_scores = attention_scores / math.sqrt(self.attention_head_size)
            attention_probs = nn.Softmax(dim=-1)(attention_scores)

        if return_probs:
            return attention_scores

        if fig_base_name:
            assert manual_attention_idx is None, manual_attention_idx
            assert attention_probs.size()[0] == 1, attention_probs.size()
            assert fig_base_name.endswith('1') or fig_base_name.endswith('2'), fig_base_name
            attention_heatmap = attention_probs.cpu().detach().numpy()[0]
            # Example base name: './heatmap/songname_1'

            # Dump attention heatmap of each attention head
            for head_idx, att in enumerate(attention_heatmap):
                att = np.sum(att, axis=0) if fig_base_name.endswith('1') else att[0]
                np.save(fig_base_name + '_{}.npy'.format(head_idx), att)
                save_heatmap_fig(att, fig_base_name + '_{}.png'.format(head_idx))

            # Dump average heatmap
            att = np.sum(attention_heatmap, axis=0)
            att = np.mean(att, axis=0) if fig_base_name.endswith('1') else att[0]
            np.save(fig_base_name + '.npy', att)
            save_heatmap_fig(att, fig_base_name + '.png'.format(head_idx))

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, manual_attention_probs=None, manual_attention_idx=None, fig_base_name=None, return_probs=False):
        if return_probs:
            return self.self(input_tensor, return_probs=return_probs)
        self_output = self.self(input_tensor, manual_attention_probs, manual_attention_idx, fig_base_name=fig_base_name)
        attention_output = self.output(self_output, input_tensor)
        return attention_output


class BertLayer(nn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, manual_attention_probs=None, manual_attention_idx=None, fig_base_name=None, return_probs=False):
        if return_probs:
            return self.attention(hidden_states, return_probs=return_probs)
        attention_output = self.attention(hidden_states, manual_attention_probs, manual_attention_idx, fig_base_name=fig_base_name)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BertEncoder(nn.Module):
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        layer = BertLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, manual_attention_probs=None, output_all_encoded_layers=True,
                idx=0, manual_attention_idx=None, fig_base_name=None, return_probs=False):
        all_encoder_layers = []
        for cur_idx, layer_module in enumerate(self.layer):
            if (cur_idx + 1) == idx:
                if return_probs:
                    return layer_module(hidden_states, return_probs=return_probs)
                hidden_states = layer_module(hidden_states, manual_attention_probs, manual_attention_idx, fig_base_name=None)
            else:
                base_name = '{}_{}'.format(fig_base_name, cur_idx + 1) if fig_base_name else None
                hidden_states = layer_module(hidden_states, manual_attention_probs=None, fig_base_name=base_name)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


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
        self.embedding = BertEmbeddings(bert_config)

        # Bert encoder
        self.encoder = BertEncoder(bert_config)

        # Bert pooler
        self.pooler = BertPooler(bert_config)

    def forward(self, x, manual_attention_probs=None, idx=0, manual_attention_idx=None, fig_base_name=None, return_probs=False):
        x = self.embedding(x)
        if return_probs:
            return self.encoder(x, idx=idx, return_probs=return_probs)
        encoded_layers = self.encoder(x, manual_attention_probs=manual_attention_probs, idx=idx, manual_attention_idx=manual_attention_idx, fig_base_name=fig_base_name)
        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output)
        return pooled_output


class MTSA(nn.Module):
    def __init__(self,
                 attention_channels=512,
                 attention_layers=2,
                 attention_heads=8,
                 attention_length=271,
                 num_class=50,
                 attention_dropout=0.1,
                 fc_dropout=0.5,
                 is_cuda=True):
        super(MTSA, self).__init__()
        self.is_cuda = is_cuda

        # Configuration
        config = Config(back_end='won',
                        conv_channels=8,
                        attention_channels=attention_channels,
                        attention_layers=attention_layers,
                        attention_heads=attention_heads,
                        attention_length=attention_length,
                        num_class=num_class,
                        batch_size=1,
                        attention_dropout=attention_dropout,
                        fc_dropout=fc_dropout,
                        is_cuda=is_cuda)

        # Front-end
        self.front_end = PonsConvModule(config)

        # Back-end
        self.back_end = AttentionModule(config)

        # Output layer
        self.dropout = nn.Dropout(fc_dropout)
        self.classifier = nn.Linear(attention_channels, num_class)

        # Sigmoid
        self.activation = nn.Sigmoid()

        # [CLS] vector
        self.vec_cls = self.get_cls(config.batch_size, config.attention_channels, config.is_cuda)

    def get_cls(self, batch, channel, is_cuda):
        np.random.seed(0)
        single_cls = torch.Tensor(np.random.random((1, channel)))
        vec_cls = torch.cat([single_cls for _ in range(32)], dim=0)
        vec_cls = vec_cls.unsqueeze(1)
        return vec_cls

    def append_cls(self, x):
        batch, _, _ = x.size()
        part_vec_cls = self.vec_cls[:batch].clone()
        part_vec_cls = part_vec_cls.to(x.device)
        return torch.cat([part_vec_cls, x], dim=1)

    def forward(self, x, idx=0, manual_attention_probs=None, manual_attention_idx=None, fig_base_name=None, return_probs=False):
        x = x.unsqueeze(1)

        # Front-end
        out = self.front_end(x)
        out = out.permute(0, 2, 1)
        out = self.append_cls(out)
        if return_probs:
            return self.back_end(out, idx=idx, return_probs=return_probs)
        out = self.back_end(out, manual_attention_probs=manual_attention_probs, idx=idx, manual_attention_idx=manual_attention_idx, fig_base_name=fig_base_name)

        # Dense
        out = self.dropout(out)
        logits = self.activation(self.classifier(out))

        return logits


class ManualAttentionVisualizer:
    def __init__(self, model, input_length=257, num_classes=50):
        self.model = model
        self.input_length = input_length
        self.num_classes = num_classes

    def get_manual_attention_t(self, x, layer_idx, t):
        return self.model(x, idx=layer_idx, manual_attention_idx=t)

    def get_manual_attention(self, x, layer_idx):
        logits = {}
        for t in range(self.input_length):
            _logit = self.get_manual_attention_t(x, layer_idx, t)
            _logit = _logit[0].cpu().detach().numpy()
            for cls_, v in enumerate(_logit):
                logits.setdefault(cls_, []).append(v)
        return {k: np.array(v) for k, v in logits.items()}
