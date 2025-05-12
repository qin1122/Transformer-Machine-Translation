#!/usr/bin/env python
# encoding: utf-8

import json
import numpy as np
import random
import torch
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from models.vanilla_transformer import *
from util import *
import time
from tqdm import tqdm
from datasets import TranslationDataset_se, TranslationDataset, TranslationDataset_old, collate_fn
from decode_sentence import decode_sentencepiece_ids, decode_chinese_ids
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction


def train(model, device, writer, train_path, test_path, val_path, cn_model_path, en_model_path, vocab_size: list, test_batch_size=32, batch_size=32, max_steps=100000, learning_rate=0.001, pad_idx=3, output_path='./results'):
    # 创建数据集和DataLoader
    train_dataset = TranslationDataset_se(
        train_path, cn_model_path, en_model_path)
    val_dataset = TranslationDataset_se(val_path, cn_model_path, en_model_path)
    test_dataset = TranslationDataset_se(
        test_path, cn_model_path, en_model_path)

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(
        val_dataset, batch_size=test_batch_size, shuffle=False, collate_fn=collate_fn)
    test_dataloader = DataLoader(
        test_dataset, batch_size=test_batch_size, shuffle=False, collate_fn=collate_fn)

    # train_dataset = TranslationDataset_old(train_path[0], train_path[1])
    # val_dataset = TranslationDataset_old(val_path[0], val_path[1])
    # test_dataset = TranslationDataset_old(test_path[0], test_path[1])
    # train_dataset = TranslationDataset(train_path, cn_model_path)
    # val_dataset = TranslationDataset(val_path, cn_model_path)
    # test_dataset = TranslationDataset(test_path, cn_model_path)

    # train_dataloader = DataLoader(
    #     train_dataset, batch_size=batch_size, shuffle=True)
    # val_dataloader = DataLoader(
    #     val_dataset, batch_size=test_batch_size, shuffle=False)
    # test_dataloader = DataLoader(
    #     test_dataset, batch_size=test_batch_size, shuffle=False)

    log_interval = 100
    val_interval = 500
    test_interval = 500
    accumulate_interval = 8

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss(
        ignore_index=pad_idx, label_smoothing=0.1)  # 忽略pad token的损失
    optimizer = optim.Adam(
        model.parameters(), lr=learning_rate, betas=[0.9, 0.98])
    scheduler = InverseSquareRootScheduler(
        optimizer, warmup_init_lr=1e-7, warmup_end_lr=learning_rate, warmup_steps=2000)

    step = 0
    total_loss = 0.0
    max_test_bleu = 0.0

    print("----------Start Training----------")

    with tqdm(total=max_steps, desc=f"Training", ncols=100) as pbar:
        while step < max_steps:
            for src, tgt in train_dataloader:
                if step >= max_steps:
                    break

                model.train()

                tgt_input = tgt[:, :-1].to(device)
                tgt_target = tgt[:, 1:].to(device)

                optimizer.zero_grad()
                output = model(src.to(device), tgt_input)

                output = output.contiguous().view(-1, vocab_size[1])
                tgt_target = tgt_target.contiguous().view(-1)

                loss = criterion(output, tgt_target)
                total_loss += loss.item()
                loss.backward()
                step += 1

                if step % accumulate_interval == 0:
                    optimizer.step()
                    # 注意：scheduler也跟step保持一致
                    scheduler.step(step//accumulate_interval)

                # tqdm实时更新 loss
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "lr": f"{scheduler.get_lr():.6f}"
                })
                pbar.update(1)

                if step % log_interval == 0:
                    avg_loss = total_loss / log_interval
                    writer.log({
                        "train_loss": avg_loss,
                        "learning_rate": scheduler.get_lr()
                    }, step=step)
                    total_loss = 0.0

                if step % val_interval == 0:
                    eval_loss = evaluate_loss(
                        model, val_dataloader, criterion, device)
                    eval_bleu = evaluate_bleu(
                        model, val_dataloader, device, pad_idx=pad_idx)
                    writer.log({
                        "eval_loss": eval_loss,
                        "eval_bleu_score": eval_bleu
                    }, step=step)

                if step % test_interval == 0:
                    test_bleu = test_bleuscore(
                        model, test_dataloader, device, max_test_bleu, pad_idx=pad_idx, output_p=output_path)
                    if max_test_bleu < test_bleu:
                        max_test_bleu = test_bleu
                    writer.log({
                        "test_bleu_score": test_bleu,
                        "max_bleu_score": max_test_bleu
                    }, step=step)

    print("----------Finish Training----------")

    # 存储模型参数
    torch.save(model.state_dict(), os.path.join(
        output_path, "vanilla_transformer.pth"))
    # print(f"模型参数已保存到 {os.path.join(output_path,"vanilla_transformer.pth")}")


def evaluate_bleu(model, val_dataloader, device, pad_idx):
    model.eval()
    bos_idx = 1
    eos_idx = 2
    total_bleu_score = 0.0

    print("\n")
    print("----------Start Validating BLEU Score----------")

    all_hypotheses = []
    all_references = []
    with tqdm(total=len(val_dataloader), desc=f"Validating_BLEU", ncols=100) as pbar:
        with torch.no_grad():
            for src, tgt in val_dataloader:
                tgt_input = tgt[:, :-1].to(device)
                tgt_target = tgt[:, 1:].to(device)
                src = src.to(device)

                output = model.greedy_decode(
                    src, max_len=50, bos_idx=bos_idx, eos_idx=eos_idx)

                for pred, truth in zip(output, tgt_target):
                    pred = remove_pad(pred, pad_idx=pad_idx)
                    truth = remove_pad(truth.tolist(), pad_idx=pad_idx)

                    pred_str = [str(tok) for tok in pred]
                    truth_str = [str(tok) for tok in truth]

                    all_hypotheses.append(pred_str)
                    all_references.append([truth_str])

                    # bleu_score.append(compute_bleu(
                    #     pred, truth, max_order=4, pad_idx=pad_idx))
                # total_bleu_score += np.mean(bleu_score)

                # # tqdm实时更新 loss
                # pbar.set_postfix({
                #     "bleu score": f"{np.mean(bleu_score):.2f}"
                # })
                pbar.update(1)

    smooth = SmoothingFunction().method4
    bleu_score = corpus_bleu(all_references, all_hypotheses, weights=(
        0.25, 0.25, 0.25, 0.25), smoothing_function=smooth)
    # avg_bleu = total_bleu_score / len(val_dataloader)
    print("\n")
    print("----------Finish Validating BLEU Score----------")
    print(f"BLEU score: {bleu_score*100:.2f}")
    return bleu_score*100


def evaluate_loss(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    total_tokens = 0

    print("\n")
    print("----------Start Validating Loss----------")

    with tqdm(total=len(dataloader), desc=f"Validating_Loss", ncols=100) as pbar:
        with torch.no_grad():
            for src, tgt in dataloader:
                src = src.to(device)
                tgt = tgt.to(device)

                tgt_input = tgt[:, :-1]
                tgt_target = tgt[:, 1:]

                output = model(src, tgt_input)  # [B, L, V]
                output = output.reshape(-1, output.shape[-1])  # [B*L, V]
                tgt_target = tgt_target.reshape(-1)  # [B*L]

                loss = criterion(output, tgt_target)
                total_loss += loss.item() * (tgt_target != 0).sum().item()
                total_tokens += (tgt_target != 0).sum().item()

                # tqdm实时更新 loss
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}"
                })
                pbar.update(1)

    avg_loss = total_loss / total_tokens
    print("\n")
    print("----------Finished Validating Loss----------")
    return avg_loss


def test_bleuscore(model, test_dataloader, device, max_test_bleu, pad_idx, output_p):
    model.eval()
    bos_idx = 1
    eos_idx = 2
    total_bleu_score = 0.0
    outputs = []

    print("\n")
    print("----------Start Testing BLEU Score----------")

    all_hypotheses = []
    all_references = []
    with tqdm(total=len(test_dataloader), desc=f"Testing_BLEU", ncols=100) as pbar:
        with torch.no_grad():
            for src, tgt in test_dataloader:
                tgt_input = tgt[:, :-1].to(device)
                tgt_target = tgt[:, 1:].to(device)
                src = src.to(device)

                output = model.greedy_decode(
                    src, max_len=50, bos_idx=bos_idx, eos_idx=eos_idx)

                for pred, truth in zip(output, tgt_target):
                    outputs.append(pred)
                    pred = remove_pad(pred, pad_idx=pad_idx)
                    truth = remove_pad(truth.tolist(), pad_idx=pad_idx)

                    pred_str = [str(tok) for tok in pred]
                    truth_str = [str(tok) for tok in truth]

                    all_hypotheses.append(pred_str)
                    all_references.append([truth_str])
                #     outputs.append(pred)
                #     bleu_score.append(compute_bleu(
                #         pred, truth, max_order=4, pad_idx=pad_idx))
                # total_bleu_score += np.mean(bleu_score)

                # # tqdm实时更新 loss
                # pbar.set_postfix({
                #     "bleu score": f"{np.mean(bleu_score):.2f}"
                # })
                pbar.update(1)

    # avg_bleu = total_bleu_score / len(test_dataloader)
    smooth = SmoothingFunction().method4
    bleu_score = corpus_bleu(all_references, all_hypotheses, weights=(
        0.25, 0.25, 0.25, 0.25), smoothing_function=smooth)
    if max_test_bleu < bleu_score*100:
        output_path = os.path.join(output_p, "output_sentence_seperate.txt")
        int2token_path = '/root/Homeworks/NLP/HW_Transformer/cmn-eng-simple/int2word_cn.json'
        clear_file_if_not_empty(output_path)
        for item in outputs:
            # decode_sentencepiece_ids(item, output_path=output_path)
            decode_chinese_ids(item, output_path, int2token_path)
    print("\n")
    print("----------Finish Testing BLEU Score----------")
    print(f"BLEU score: {bleu_score*100:.2f}")
    return bleu_score*100
