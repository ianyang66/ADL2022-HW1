from typing import List, Dict

import torch
from torch.utils.data import Dataset

from utils import Vocab
import re


class SeqClsDataset(Dataset):
    def __init__(
        self,
        data: List[Dict],
        vocab: Vocab,
        label_mapping: Dict[str, int],
        max_len: int,
    ):
        self.data = data
        self.vocab = vocab
        self.label_mapping = label_mapping
        self._idx2label = {idx: intent for intent, idx in self.label_mapping.items()}
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        instance = self.data[index]
        return instance

    @property
    def num_classes(self) -> int:
        return len(self.label_mapping)

    def collate_fn(self, samples: List[Dict]) -> Dict:
        # TODO: implement collate_fn
        batch = {"text": [], "intent": [], "id": []}
        for sample in samples:
            batch["id"].append(sample["id"])
            batch["text"] += [(re.sub(r"(\w)([^a-zA-Z0-9 ])", r"\1 \2",sample["text"]).split())]
            #batch["text"].append(re.sub(r"(\w)([^a-zA-Z0-9 ])", r"\1 \2", sample["text"]).split(" "))
            #print(batch["text"])
            try:
                batch["intent"].append(self.label_mapping[sample["intent"]])
            except:
                pass
        batch['text'] = self.vocab.encode_batch(batch["text"], self.max_len)
        batch['text'] = torch.LongTensor(batch["text"])
        #batch["text"] = torch.LongTensor(self.vocab.encode_batch(batch["text"], self.max_len))
        try:
            batch["intent"] = torch.LongTensor(batch["intent"])
        except:
            pass
        return batch

    def label2idx(self, label: str):
        return self.label_mapping[label]

    def idx2label(self, idx: int):
        return self._idx2label[idx]

def pad_to_len(seqs: List[List[int]], to_len: int, padding: int) -> List[List[int]]:
        paddeds = [seq[:to_len] + [padding] * max(0, to_len - len(seq)) for seq in seqs]
        return paddeds

class SeqSlotDataset(Dataset):
    def __init__(
        self,
        data: List[Dict],
        vocab: Vocab,
        label_mapping: Dict[str, int],
        max_len: int,
    ):
        self.data = data
        self.vocab = vocab
        self.label_mapping = label_mapping
        self._idx2label = {idx: intent for intent, idx in self.label_mapping.items()}
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        instance = self.data[index]
        return instance

    @property
    def num_classes(self) -> int:
        return len(self.label_mapping)

    

    def collate_fn(self, samples: List[Dict]) -> Dict:
        # TODO: implement collate_fn
        results = {"tokens": [], "tags": [], "id": []}
        for data in samples:
            results["id"].append(data["id"])
            results["tokens"].append(data["tokens"])
            try:
                results["tags"].append(list(map(lambda x: self.label_mapping[x], data["tags"])))
            except:
                pass

        results["tokens"] = torch.LongTensor(self.vocab.encode_batch(results["tokens"], self.max_len))
        try:
            results["tags"] = torch.LongTensor(pad_to_len(results["tags"],self.max_len, 9))
        except:
            pass
        return results

    def label2idx(self, label: str):
        return self.label_mapping[label]

    def idx2label(self, idx: int):
        return self._idx2label[idx]