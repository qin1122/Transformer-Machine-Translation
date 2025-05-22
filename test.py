import torch
import json
import yaml
import argparse
from models.vanilla_transformer import Vanilla_TransformerModel


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_yaml(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def tokenize_en(sentence: str):
    return sentence.lower().strip().split()


def encode(tokens, word2int, unk_token='<unk>'):
    return [word2int.get(token, word2int.get(unk_token, 0)) for token in tokens]


def decode(ids, int2word, eos_idx=2):
    tokens = []
    for i in ids:
        if i == eos_idx:
            break
        tokens.append(int2word.get(str(i), ''))
    return ''.join(tokens)


def translate_sentence(cfg, sentence: str, model, word2int_en, int2word_cn, device):
    tokens = tokenize_en(sentence)
    src_ids = encode(tokens, word2int_en)
    src_tensor = torch.LongTensor([src_ids]).to(device)

    with torch.no_grad():
        output_ids = model.greedy_decode(src_tensor, max_len=cfg['max_len'],
                                         bos_idx=cfg['bos_idx'], eos_idx=cfg['eos_idx'])[0]

    return decode(output_ids, int2word_cn, eos_idx=cfg['eos_idx'])


def load_model(cfg, device):
    word2int_en = load_json(cfg['word2int_en'])
    int2word_cn = load_json(cfg['int2word_cn'])

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

    model.load_state_dict(torch.load(cfg['model_path'], map_location=device))
    model.to(device)
    model.eval()
    return model, word2int_en, int2word_cn


def main():
    parser = argparse.ArgumentParser(
        description="Translate English to Chinese using Transformer.")
    parser.add_argument('--config', type=str,
                        default='config.yaml', help='Path to YAML config file')
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    device = cfg['device'] if torch.cuda.is_available() else 'cpu'

    model, word2int_en, int2word_cn = load_model(cfg, device)

    print("请输入英文句子，输入 'exit' 退出：")
    while True:
        en_sentence = input(">>> ")
        if en_sentence.strip().lower() == 'exit':
            break
        zh_translation = translate_sentence(
            cfg, en_sentence, model, word2int_en, int2word_cn, device)
        print("翻译结果:", zh_translation)


if __name__ == '__main__':
    main()
