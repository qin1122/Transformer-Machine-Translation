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
import yaml
import argparse


def load_yaml(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Translate English to Chinese using a trained Transformer model.")

    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to YAML config file')
    args = parser.parse_args()

    cfg = load_yaml(args.config)

    # 设置随机种子保证可复现
    set_seed(cfg['seed'])

    if cfg['norm_first']:
        name = f'Pre_norm+{cfg['dataset_type']}+n_heads_{cfg['nhead']}+{cfg['num_encoder_layers']}+pe_{cfg['use_pe']}+ls_{cfg['use_ls']}'

    wandb.init(
        project="Transformer Translation Experiments",
        name=name,
        config=cfg
    )

    # <UNK>: 0, <BOS>: 1, <EOS>: 2, <PAD>:3
    # train_path = [
    #     '/root/Homeworks/NLP/HW_Transformer/my_transformer/dataset/tokenized_dataset_old/train_en.txt', '/root/Homeworks/NLP/HW_Transformer/my_transformer/dataset/tokenized_dataset_old/train_cn.txt']
    # test_path = [
    #     '/root/Homeworks/NLP/HW_Transformer/my_transformer/dataset/tokenized_dataset_old/test_en.txt', '/root/Homeworks/NLP/HW_Transformer/my_transformer/dataset/tokenized_dataset_old/test_cn.txt']
    # val_path = [
    #     '/root/Homeworks/NLP/HW_Transformer/my_transformer/dataset/tokenized_dataset_old/val_en.txt', '/root/Homeworks/NLP/HW_Transformer/my_transformer/dataset/tokenized_dataset_old/val_cn.txt']
    output_path = cfg['output_path']
    os.makedirs(output_path, exist_ok=True)

    # 初始化模型
    model = Vanilla_TransformerModel(
        vocab_size=cfg['vocab_size'],
        d_model=cfg['d_model'],
        nhead=cfg['nhead'],
        num_encoder_layers=cfg['num_encoder_layers'],
        num_decoder_layers=cfg['num_decoder_layers'],
        dropout=cfg['dropout'],
        pad_idx=cfg['pad_idx'],
        combine_split=False,
        pe=cfg['use_pe'],
        norm_first=cfg['norm_first']
    )

    device = torch.device(
        cfg['device'] if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 调用训练函数
    train(
        args=cfg,
        model=model,
        device=device,
        writer=wandb
    )


if __name__ == "__main__":
    main()
