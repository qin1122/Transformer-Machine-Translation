#!/usr/bin/env python
# encoding: utf-8

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from util import *


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  # [max_len, d_model]
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数位置
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数位置

        pe = pe.unsqueeze(1)  # [max_len, 1, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [seq_len, batch_size, d_model]
        x = x + self.pe[:x.size(0)]
        return x


class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Parameter(torch.zeros(max_len, d_model))  # 可学习参数
        nn.init.normal_(self.pe, mean=0, std=0.02)  # 初始化方式可以自由调整

    def forward(self, x):
        # x: [seq_len, batch_size, d_model]
        seq_len = x.size(0)
        x = x + self.pe[:seq_len].unsqueeze(1)  # [seq_len, 1, d_model]
        return self.dropout(x)


class Vanilla_TransformerModel(nn.Module):
    def __init__(self, vocab_size: list, d_model=512, nhead=8,
                 num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=4096,
                 dropout=0.1, pad_idx=0, combine_split=False, pe=True, norm_first=False):
        super().__init__()

        self.d_model = d_model
        self.pad_idx = pad_idx
        self.combine_split = combine_split
        self.pe = pe

        if self.combine_split:
            self.embedding = nn.Embedding(
                vocab_size[0], d_model, padding_idx=pad_idx)
        else:
            self.src_embedding = nn.Embedding(
                vocab_size[0], d_model, padding_idx=pad_idx
            )
            self.tgt_embedding = nn.Embedding(
                vocab_size[1], d_model, padding_idx=pad_idx
            )
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, norm_first=norm_first)
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model, nhead, dim_feedforward, dropout, norm_first=norm_first)
        self.decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_decoder_layers)

        self.output_proj = nn.Linear(d_model, vocab_size[1])

    def make_pad_mask(self, seq, pad_idx):
        # seq: [batch_size, seq_len]
        return (seq == pad_idx)

    def generate_square_subsequent_mask(self, sz):
        return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)

    def forward(self, src, tgt):
        # src: [batch, src_len], tgt: [batch, tgt_len]
        src_mask = None
        tgt_mask = self.generate_square_subsequent_mask(
            tgt.size(1)).to(tgt.device)

        src_key_padding_mask = self.make_pad_mask(src, self.pad_idx)
        tgt_key_padding_mask = self.make_pad_mask(tgt, self.pad_idx)

        if self.combine_split:
            src_emb = self.embedding(
                src) * math.sqrt(self.d_model)  # [B, L, D]
            tgt_emb = self.embedding(tgt) * math.sqrt(self.d_model)
        else:
            src_emb = self.src_embedding(
                src) * math.sqrt(self.d_model)  # [B, L, D]
            tgt_emb = self.tgt_embedding(tgt) * math.sqrt(self.d_model)

        if self.pe:
            src_emb = self.pos_encoder(src_emb.transpose(0, 1))  # [L, B, D]
            tgt_emb = self.pos_encoder(tgt_emb.transpose(0, 1))
            # print(src_emb[:, 0, :])
        else:
            src_emb = src_emb.transpose(0, 1)
            tgt_emb = tgt_emb.transpose(0, 1)

        memory = self.encoder(src_emb, mask=src_mask,
                              src_key_padding_mask=src_key_padding_mask)
        output = self.decoder(tgt_emb, memory,
                              tgt_mask=tgt_mask,
                              tgt_key_padding_mask=tgt_key_padding_mask,
                              memory_key_padding_mask=src_key_padding_mask)

        output = self.output_proj(output)  # [tgt_len, batch_size, vocab_size]
        return output.transpose(0, 1)  # [batch_size, tgt_len, vocab_size]

    def greedy_decode(self, src, bos_idx, eos_idx, max_len=50):
        """
        src: [B, src_len]
        返回: List[List[int]]，每个元素是生成的 token id 序列
        """
        device = src.device
        batch_size = src.size(0)

        with torch.no_grad():
            # 编码器部分
            src_mask = None
            src_key_padding_mask = self.make_pad_mask(
                src, self.pad_idx)  # [B, src_len]
            if self.combine_split:
                src_emb = self.embedding(
                    src) * math.sqrt(self.d_model)
            else:
                src_emb = self.src_embedding(
                    src) * math.sqrt(self.d_model)
            if self.pe:
                src_emb = self.pos_encoder(
                    src_emb.transpose(0, 1))            # [L, B, D]
            else:
                src_emb = src_emb.transpose(0, 1)
            memory = self.encoder(src_emb, mask=src_mask,
                                  src_key_padding_mask=src_key_padding_mask)  # [L, B, D]

            # 初始化 tgt: [B, 1]
            ys = torch.full((batch_size, 1), bos_idx,
                            dtype=torch.long, device=device)  # 初始全部是 <bos>
            finished = torch.zeros(batch_size, dtype=torch.bool,
                                   device=device)  # 标记哪些已经生成 <eos>

            for _ in range(max_len):
                tgt_mask = self.generate_square_subsequent_mask(
                    ys.size(1)).to(device)
                tgt_key_padding_mask = self.make_pad_mask(
                    ys, self.pad_idx)  # [B, tgt_len]

                if self.combine_split:
                    tgt_emb = self.embedding(
                        ys) * math.sqrt(self.d_model)
                else:
                    tgt_emb = self.tgt_embedding(
                        ys) * math.sqrt(self.d_model)

                if self.pe:
                    tgt_emb = self.pos_encoder(
                        tgt_emb.transpose(0, 1))  # [tgt_len, B, D]
                else:
                    tgt_emb = tgt_emb.transpose(0, 1)

                output = self.decoder(tgt_emb, memory,
                                      tgt_mask=tgt_mask,
                                      memory_key_padding_mask=src_key_padding_mask,
                                      tgt_key_padding_mask=tgt_key_padding_mask)
                output = self.output_proj(output)  # [tgt_len, B, vocab_size]
                next_token = output[-1].argmax(-1)  # [B]

                # 将还未结束的句子拼接上新生成的 token
                ys = torch.cat([ys, next_token.unsqueeze(1)],
                               dim=1)  # [B, L+1]

                # 如果某个句子已经生成了 eos，就不再更新它
                finished |= (next_token == eos_idx)
                if finished.all():
                    break

        # 转换成 Python 列表，按每个句子处理
        results = []
        for i in range(batch_size):
            tokens = ys[i].tolist()
            if eos_idx in tokens:
                eos_pos = tokens.index(eos_idx)
                tokens = tokens[:eos_pos+1]
            results.append(tokens)

        return results  # List[List[int]]
