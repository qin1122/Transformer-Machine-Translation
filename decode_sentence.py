import json
import sentencepiece as spm


def decode_sentencepiece_ids(token_ids, output_path, pad_id=3, bos_id=1, eos_id=2):
    """
    从 token_ids(含padding)还原为正常句子
    参数：
        token_ids: list[int]，一个样本的 token id 序列(含padding)
        sp_model_path: SentencePiece 模型路径
        pad_id, bos_id, eos_id: 对应的特殊token id
    返回：
        str: 解码后的句子
    """
    sp_model_path = '/root/Homeworks/NLP/HW_Transformer/my_transformer/dataset/seperate_bpe_preprocess/cn_bpe.model'

    sp = spm.SentencePieceProcessor()
    sp.load(sp_model_path)

    # 去除 padding、BOS、EOS
    clean_ids = [i for i in token_ids if i not in (pad_id, bos_id, eos_id)]

    # 解码：token ids → token pieces → sentence
    tokens = sp.id_to_piece(clean_ids)
    sentence = ''.join(tokens).replace('▁', '').strip()

    with open(output_path, 'a', encoding='utf-8') as f:
        f.write(sentence+'\n')
    # print(f"[✓] 已追加写入1条句子到 {output_path}")

    return sentence


def decode_chinese_ids(id_sequence, output_path, int2token_path, skip_special_tokens=True):
    """
    将中文 token id 序列转换为句子。

    参数：
        id_sequence (list[int]): 编码后的 token ID 序列
        int2token_path (str): 指向 int2token.json 的路径
        skip_special_tokens (bool): 是否跳过特殊符号

    返回：
        str: 解码后的中文句子
    """
    # 加载 int2token 映射字典
    with open(int2token_path, 'r', encoding='utf-8') as f:
        int2token = json.load(f)

    # 默认跳过这些特殊符号
    special_tokens = {'<PAD>', '<BOS>',
                      '<EOS>'} if skip_special_tokens else set()

    tokens = [
        int2token.get(str(idx), '')
        for idx in id_sequence
        if int2token.get(str(idx), '') not in special_tokens
    ]
    sentence = ''.join(tokens)
    with open(output_path, 'a', encoding='utf-8') as f:
        f.write(sentence+'\n')

    return sentence
