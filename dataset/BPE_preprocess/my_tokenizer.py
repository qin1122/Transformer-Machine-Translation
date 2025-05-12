import sentencepiece as spm
import os
import re
import json


def extract_en_cn(input_path, en_path, cn_path):
    with open(input_path, 'r', encoding='utf-8') as fin, \
            open(en_path, 'w', encoding='utf-8') as fen, \
            open(cn_path, 'w', encoding='utf-8') as fcn:
        for line in fin:
            parts = re.split(r'\t+', line.strip())
            if len(parts) < 2:
                continue
            en_sent, cn_sent = parts[0], parts[1]
            fen.write(en_sent.strip() + '\n')
            fcn.write(cn_sent.strip() + '\n')
    print(f"[✓] 提取完成：英文 → {en_path}，中文 → {cn_path}")


def merge_files(en_path, cn_path, out_path):
    with open(out_path, 'w', encoding='utf-8') as fout:
        for path in [en_path, cn_path]:
            with open(path, 'r', encoding='utf-8') as fin:
                fout.writelines(fin.readlines())
    print(f"[✓] 合并文件：{out_path}")


def train_bpe_model(train_path, model_prefix='bpe', vocab_size=8000, character_coverage=1.0):
    spm.SentencePieceTrainer.train(
        input=train_path,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        character_coverage=1.0,  # 中文需要设为 1.0
        model_type='bpe',
        user_defined_symbols=['<pad>']
    )
    print(f"[✓] BPE 模型训练完成：{model_prefix}.model")


def apply_bpe(sp_model_path, input_path, output_path):
    sp = spm.SentencePieceProcessor()
    sp.load(sp_model_path)
    with open(input_path, 'r', encoding='utf-8') as fin, \
            open(output_path, 'w', encoding='utf-8') as fout:
        for line in fin:
            line = line.strip()
            if line:
                pieces = sp.encode(line, out_type=str)
                fout.write(' '.join(pieces) + '\n')
    print(f"[✓] BPE 分词完成：{input_path} → {output_path}")


def save_vocab_mappings(model_path='bpe.model', vocab_path='int2token.json', token2id_path='token2int.json'):
    sp = spm.SentencePieceProcessor()
    sp.load(model_path)

    vocab = {}
    token2id = {}

    for i in range(sp.get_piece_size()):
        token = sp.id_to_piece(i)
        vocab[i] = token
        token2id[token] = i

    with open(vocab_path, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    with open(token2id_path, 'w', encoding='utf-8') as f:
        json.dump(token2id, f, ensure_ascii=False, indent=2)

    print(f"[✓] id → token 映射已保存到：{vocab_path}")
    print(f"[✓] token → id 映射已保存到：{token2id_path}")


def main():
    # 文件路径配置
    cmn_file = '/root/Homeworks/NLP/HW_Transformer/my_transformer/dataset/BPE_preprocess/cmn.txt'
    en_file = '/root/Homeworks/NLP/HW_Transformer/my_transformer/dataset/BPE_preprocess/en.txt'
    cn_file = '/root/Homeworks/NLP/HW_Transformer/my_transformer/dataset/BPE_preprocess/cn.txt'
    vocab_size = 5000

    # 步骤 1：提取中英文句子
    extract_en_cn(cmn_file, en_file, cn_file)

    # 步骤 2：训练 BPE 模型
    train_bpe_model(en_file, model_prefix='en_bpe',
                    vocab_size=vocab_size, character_coverage=0.9995)
    train_bpe_model(cn_file, model_prefix='cn_bpe',
                    vocab_size=vocab_size, character_coverage=1.0)

    # 步骤 3：对中英文分别进行 BPE 分词
    apply_bpe('en_bpe.model', en_file,
              os.path.join('en_bpe'+str(vocab_size)+'.txt'))
    apply_bpe('cn_bpe.model', cn_file,
              os.path.join('cn_bpe'+str(vocab_size)+'.txt'))

    save_vocab_mappings(
        model_path='en_bpe.model',
        vocab_path='int2token_en.json',
        token2id_path='token2int_en.json'
    )
    save_vocab_mappings(
        model_path='cn_bpe.model',
        vocab_path='int2token_cn.json',
        token2id_path='token2int_cn.json'
    )

    print("[✓] 所有步骤完成!")


if __name__ == "__main__":
    main()
