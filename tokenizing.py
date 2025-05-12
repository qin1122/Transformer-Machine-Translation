import json
import re
import os

DATA_DIR = '/root/Homeworks/NLP/HW_Transformer/cmn-eng-simple'
RESULT_DIR = '/root/Homeworks/NLP/HW_Transformer/my_transformer/dataset/tokenized_dataset_old'
os.makedirs(RESULT_DIR, exist_ok=True)

# 读取词表
with open(os.path.join(DATA_DIR, "word2int_en.json"), "r") as f:
    word2idx_en = json.load(f)
with open(os.path.join(DATA_DIR, "word2int_cn.json"), "r") as f:
    word2idx_cn = json.load(f)

# 定义特殊符号的 ID
PAD_ID = word2idx_en['<PAD>']
BOS_ID = word2idx_en['<BOS>']
EOS_ID = word2idx_en['<EOS>']
UNK_ID = word2idx_en['<UNK>']

# 读取文本
with open(os.path.join(DATA_DIR, "testing.txt"), "r") as f:
    lines = f.readlines()

# 编码函数


def encode_sentence(tokens, word2idx, max_len=50):
    # 将单词转换为对应的索引，找不到的单词用 <UNK> 替代
    ids = [word2idx.get(word, UNK_ID) for word in tokens]
    # 添加 BOS 和 EOS
    ids = [BOS_ID] + ids + [EOS_ID]
    # 如果长度不足 max_len，使用 PAD 填充
    if len(ids) < max_len:
        ids += [PAD_ID] * (max_len - len(ids))
    else:
        print("Sentence length exceeds max_len, truncating.")
        ids = ids[:max_len]  # 如果超过 max_len，截断
    return ids


# 编码所有句子
en_encoded = []
cn_encoded = []
max_len = 50  # 设置最大长度

for line in lines:
    # 分割英文和中文句子
    en_sentence, cn_sentence = line.strip().split('\t')

    # 将英文和中文句子分别分词并编码为索引
    en_tokens = en_sentence.split()  # 使用空格分词
    cn_tokens = cn_sentence.split()  # 使用空格分词

    en_ids = encode_sentence(en_tokens, word2idx_en, max_len)
    cn_ids = encode_sentence(cn_tokens, word2idx_cn, max_len)

    en_encoded.append(en_ids)
    cn_encoded.append(cn_ids)

# 保存编码后的数据
with open(os.path.join(RESULT_DIR, 'test_en.txt'), "w") as f:
    json.dump(en_encoded, f)

with open(os.path.join(RESULT_DIR, 'test_cn.txt'), "w") as f:
    json.dump(cn_encoded, f)

print("编码完成，已保存索引格式的句子！")
