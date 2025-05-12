#!/usr/bin/env python
# encoding: utf-8

import os
import torch
import numpy as np
import random
from train import *
from util import *
from models.vanilla_transformer import Vanilla_TransformerModel
import wandb


def main():
    # 设置随机种子保证可复现
    set_seed(42)

    wandb.init(
        project="Transformer Translation Experiments",
        name="Pre_norm + newdataset + n_heads_8 + 16_layers + pe + ls",
        config={
            "steps": 160000,
            "batch_size": 128,
            "n_heads": 8,
            "num_encoder_layers": 16,
            "num_decoder_layers": 16,
            "use pe": True,
            "warmup_init_lr": 1e-7,
            "warmup_end_lr": 1e-4,
            "optimizer": "Adam"
        }
    )

    # <UNK>: 0, <BOS>: 1, <EOS>: 2, <PAD>:3
    train_path = '/root/Homeworks/NLP/HW_Transformer/my_transformer/dataset/seperate_bpe_preprocess/training.txt'
    test_path = '/root/Homeworks/NLP/HW_Transformer/my_transformer/dataset/seperate_bpe_preprocess/testing.txt'
    val_path = '/root/Homeworks/NLP/HW_Transformer/my_transformer/dataset/seperate_bpe_preprocess/validation.txt'
    # train_path = [
    #     '/root/Homeworks/NLP/HW_Transformer/my_transformer/dataset/tokenized_dataset_old/train_en.txt', '/root/Homeworks/NLP/HW_Transformer/my_transformer/dataset/tokenized_dataset_old/train_cn.txt']
    # test_path = [
    #     '/root/Homeworks/NLP/HW_Transformer/my_transformer/dataset/tokenized_dataset_old/test_en.txt', '/root/Homeworks/NLP/HW_Transformer/my_transformer/dataset/tokenized_dataset_old/test_cn.txt']
    # val_path = [
    #     '/root/Homeworks/NLP/HW_Transformer/my_transformer/dataset/tokenized_dataset_old/val_en.txt', '/root/Homeworks/NLP/HW_Transformer/my_transformer/dataset/tokenized_dataset_old/val_cn.txt']
    bpe_model_path = '/root/Homeworks/NLP/HW_Transformer/my_transformer/dataset/BPE_preprocess/bpe.model'
    cn_model_path = '/root/Homeworks/NLP/HW_Transformer/my_transformer/dataset/seperate_bpe_preprocess/cn_bpe.model'
    en_model_path = '/root/Homeworks/NLP/HW_Transformer/my_transformer/dataset/seperate_bpe_preprocess/en_bpe.model'
    output_path = './results/result_15'
    os.makedirs(output_path, exist_ok=True)

    # 模型参数s
    vocab_size = [5000, 5000]  # 根据你的词表大小来设定
    pad_idx = 3
    d_model = 1024
    nhead = 8
    num_encoder_layers = 16
    num_decoder_layers = 16
    dropout = 0.3

    # 初始化模型
    model = Vanilla_TransformerModel(
        vocab_size=vocab_size,
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        dropout=dropout,
        pad_idx=pad_idx,
        combine_split=False,
        pe=True,
        norm_first=True
    )

    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 调用训练函数
    train(
        model=model,
        device=device,
        writer=wandb,
        train_path=train_path,
        test_path=test_path,
        val_path=val_path,
        cn_model_path=cn_model_path,
        en_model_path=en_model_path,
        vocab_size=vocab_size,
        test_batch_size=32,
        batch_size=128,
        max_steps=160000,
        learning_rate=1e-4,
        pad_idx=pad_idx,
        output_path=output_path
    )


if __name__ == "__main__":
    main()
