#!/usr/bin/env python
# encoding: utf-8

from torch.optim import Optimizer
import torch.nn as nn
from collections import Counter
import math
import torch
import json
import numpy as np
import random
import os


class InverseSquareRootScheduler:
    def __init__(self, optimizer, warmup_init_lr, warmup_end_lr, warmup_steps=4000):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.warmup_init_lr = warmup_init_lr
        self.warmup_end_lr = warmup_end_lr

        if self.warmup_init_lr < 0:
            self.warmup_init_lr = self.warmup_end_lr
        # linearly warmup for the first warmup_steps
        self.lr_step = (self.warmup_end_lr - self.warmup_init_lr) / \
            self.warmup_steps
        # then, decay prop. to the inverse square root of the update number
        self.decay_factor = self.warmup_end_lr * self.warmup_steps**0.5
        self.lr = self.warmup_init_lr
        self.set_lr(self.lr)

    def set_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def step(self, _step):
        if _step < self.warmup_steps:
            self.lr = self.warmup_init_lr+_step*self.lr_step
        else:
            self.lr = self.decay_factor*_step**-0.5
        self.set_lr(self.lr)
        return self.lr

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']


class NoamScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer: Optimizer, model_size: int, warmup_steps: int, last_epoch: int = -1):
        """
        实现 Transformer 中的 Noam 学习率调度器。

        Args:
            optimizer: PyTorch 的优化器（如 Adam）
            model_size: Transformer 的隐藏层维度（如 512）
            warmup_steps: 预热步数（如 4000）
            last_epoch: 上一轮的 epoch（默认 -1）
        """
        self.model_size = model_size
        self.warmup_steps = warmup_steps
        super(NoamScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        # step 从1开始防止除零
        step = max(self.last_epoch, 1)
        scale = self.model_size ** -0.5 * \
            min(step ** -0.5, step * self.warmup_steps ** -1.5)
        return [base_lr * scale for base_lr in self.base_lrs]


def n_gram(tokens, n):
    return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]


def remove_pad(seq, pad_idx=3):
    # 去掉 PAD（只保留第一个 pad 之前的）
    if pad_idx in seq:
        seq = seq[:seq.index(pad_idx)]

    return seq


def compute_bleu(pred: list, ref, max_order=4, pad_idx=3):
    """
    :param pred: list(1D), e.g., tensor([1, 2, 3, 4])
    :param ref:  torch.Tensor (1D), same as above
    :param max_order: maximum n-gram order (default 4)
    :return: BLEU score (float)
    """
    ref = ref.tolist()
    pred = remove_pad(pred, pad_idx=pad_idx)
    ref = remove_pad(ref, pad_idx=pad_idx)

    matches_by_order = [0] * max_order
    possible_matches_by_order = [0] * max_order

    for n in range(1, max_order + 1):
        pred_ngrams = Counter(n_gram(pred, n))
        ref_ngrams = Counter(n_gram(ref, n))

        # intersection counts matching n-grams
        overlap = pred_ngrams & ref_ngrams
        matches_by_order[n - 1] = sum(overlap.values())
        possible_matches_by_order[n - 1] = max(len(pred) - n + 1, 0)

    precisions = [
        (matches_by_order[i] / possible_matches_by_order[i]
         ) if possible_matches_by_order[i] > 0 else 0.0
        for i in range(max_order)
    ]

    # Brevity penalty
    pred_len = len(pred)
    ref_len = len(ref)
    bp = 1.0 if pred_len > ref_len else math.exp(
        1 - ref_len / pred_len) if pred_len > 0 else 0.0

    # geometric mean of n-gram precisions
    if all(p > 0 for p in precisions):
        score = bp * math.exp(sum(math.log(p+1e-8)
                              for p in precisions) / max_order)
    else:
        score = 0.0

    return score * 100  # return percentage


def load_encoded_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)  # 一次性加载整个列表
    return data  # 返回的是 List[List[int]]


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def subsequent_mask(size):
    """生成一个上三角矩阵，掩盖未来的时间步"""
    return torch.triu(torch.full((size, size), float('-inf')), diagonal=1)


def clear_file_if_not_empty(file_path):
    if os.path.isfile(file_path):
        if os.path.getsize(file_path) > 0:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.truncate(0)  # 清空文件内容
            print(f"[✓] 文件不为空，已清空：{file_path}")
        else:
            print(f"[i] 文件已为空：{file_path}")
    else:
        print(f"[✗] 文件不存在：{file_path}")
