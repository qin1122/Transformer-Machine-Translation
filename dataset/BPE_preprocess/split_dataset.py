import os
import random

en_path = 'en_bpe8000.txt'
cn_path = 'cn_bpe8000.txt'

sentences = []

with open(en_path, 'r', encoding='utf-8') as f_en, open(cn_path, 'r', encoding='utf-8') as f_cn:
    for en_line, cn_line in zip(f_en, f_cn):
        en_line = en_line.strip()
        cn_line = cn_line.strip()
        if en_line and cn_line:
            sentences.append(en_line+'\t'+cn_line)


print(f"[✓] 加载完成，共有 {len(sentences)} 个句子对。")

random.seed(2020)
random.shuffle(sentences)

with open('./training.txt', 'w') as f:
    for sentence in sentences[:18000]:
        print(sentence, file=f)

with open('./validation.txt', 'w') as f:
    for sentence in sentences[18000:18500]:
        print(sentence, file=f)

with open('./testing.txt', 'w') as f:
    for sentence in sentences[18500:]:
        print(sentence, file=f)

print('Build Dataset Down!')
