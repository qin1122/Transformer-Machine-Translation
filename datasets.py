import ast
import sentencepiece as spm
import torch
from torch.utils.data import Dataset
import json


class TranslationDataset(Dataset):
    def __init__(self, file_path, sp_model_path, add_bos=True, add_eos=True):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(sp_model_path)

        self.add_bos = add_bos
        self.add_eos = add_eos
        self.src_ids = []
        self.tgt_ids = []

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) != 2:
                    continue
                en, cn = parts
                self.src_ids.append(self.encode(en, max_len=70))
                self.tgt_ids.append(self.encode(cn, max_len=70))

    def encode(self, sentence, max_len=70):
        ids = self.sp.encode(sentence.strip(), out_type=int)
        if self.add_bos:
            ids = [self.sp.bos_id()] + ids
        if self.add_eos:
            ids = ids + [self.sp.eos_id()]

        # add padding
        if len(ids) <= max_len:
            ids += [3]*(max_len-len(ids))
        else:
            print(f"Sentence is too long, len(ids)={len(ids)}")
            ids = ids[:max_len]
        return ids

    def __len__(self):
        return len(self.src_ids)

    def __getitem__(self, idx):
        return torch.tensor(self.src_ids[idx], dtype=torch.long), \
            torch.tensor(self.tgt_ids[idx], dtype=torch.long)


class TranslationDataset_se(Dataset):
    def __init__(self, file_path, cn_model_path, en_model_path, add_bos=True, add_eos=True):
        self.sp_cn = spm.SentencePieceProcessor()
        self.sp_cn.load(cn_model_path)
        self.sp_en = spm.SentencePieceProcessor()
        self.sp_en.load(en_model_path)

        self.add_bos = add_bos
        self.add_eos = add_eos
        self.src_ids = []
        self.tgt_ids = []

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) != 2:
                    continue
                en, cn = parts
                self.src_ids.append(self.encode_en(en, max_len=70))
                self.tgt_ids.append(self.encode_cn(cn, max_len=70))

    def encode_cn(self, sentence, max_len=70):
        ids = self.sp_cn.encode(sentence.strip(), out_type=int)
        if self.add_bos:
            ids = [self.sp_cn.bos_id()] + ids
        if self.add_eos:
            ids = ids + [self.sp_cn.eos_id()]

        # add padding
        if len(ids) <= max_len:
            ids += [3]*(max_len-len(ids))
        else:
            print(f"Sentence is too long, len(ids)={len(ids)}")
            ids = ids[:max_len]
        return ids

    def encode_en(self, sentence, max_len=70):
        ids = self.sp_en.encode(sentence.strip(), out_type=int)
        if self.add_bos:
            ids = [self.sp_en.bos_id()] + ids
        if self.add_eos:
            ids = ids + [self.sp_en.eos_id()]

        # add padding
        if len(ids) <= max_len:
            ids += [3]*(max_len-len(ids))
        else:
            print(f"Sentence is too long, len(ids)={len(ids)}")
            ids = ids[:max_len]
        return ids

    def __len__(self):
        return len(self.src_ids)

    def __getitem__(self, idx):
        return torch.tensor(self.src_ids[idx], dtype=torch.long), \
            torch.tensor(self.tgt_ids[idx], dtype=torch.long)


def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    src_batch = torch.stack(src_batch, dim=0)
    tgt_batch = torch.stack(tgt_batch, dim=0)
    return src_batch, tgt_batch


class TranslationDataset_old(Dataset):
    def __init__(self, src_path, tgt_path):
        self.src_sequences = self._load_file(src_path)
        self.tgt_sequences = self._load_file(tgt_path)

        assert len(self.src_sequences) == len(self.tgt_sequences), "源和目标长度不一致"

    def _load_file(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            return ast.literal_eval(f.read())

    def __len__(self):
        return len(self.src_sequences)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.src_sequences[idx], dtype=torch.long),
            torch.tensor(self.tgt_sequences[idx], dtype=torch.long)
        )


# # 示例执行
# if __name__ == '__main__':
#     # 修改为你的文件路径
#     # 中英文分词后合并文件（\t 分隔）
#     file_path = '/root/Homeworks/NLP/HW_Transformer/my_transformer/dataset/BPE_preprocess/validation.txt'
#     # SentencePiece 模型路径
#     bpe_model_path = '/root/Homeworks/NLP/HW_Transformer/my_transformer/dataset/BPE_preprocess/bpe.model'
#     dataset = TranslationDataset(file_path, bpe_model_path)

#     print(f"[✓] 加载完成，样本数：{len(dataset)}")
