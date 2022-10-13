import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import trange, tqdm
import numpy as np
import random

from model_seqSlotClassifier import SeqSlotLSTMClassifier, SeqSlotClassifier
from utils import Vocab
from dataset import SeqSlotDataset
from seqeval.metrics import accuracy_score
from seqeval.metrics import classification_report
from seqeval.metrics import f1_score
from seqeval.scheme import IOB2


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    tag_idx_path = args.cache_dir / "tag2idx.json"
    tag2idx: Dict[str, int] = json.loads(tag_idx_path.read_text())

    data = json.loads(args.test_file.read_text())
    dataset = SeqSlotDataset(data, vocab, tag2idx, args.max_len)
    # crecate DataLoader for test dataset
    test_loader = DataLoader(dataset, args.batch_size, shuffle=False, collate_fn=dataset.collate_fn)

    embeddings = torch.load(args.cache_dir / "embeddings.pt")

    """model = SeqSlotClassifier(
        embeddings,
        args.hidden_size,
        args.num_layers,
        args.dropout,
        args.bidirectional,
        num_class = 10,
    ).to(args.device)
    """
    model = SeqSlotLSTMClassifier(
        embeddings,
        args.hidden_size,
        args.num_layers,
        args.dropout,
        args.bidirectional,
        num_class = 10,
    ).to(args.device)
    model.eval()

    ckpt = torch.load(args.ckpt_path)
    # load weights into model
    model.load_state_dict(ckpt)

    all_pred = []
    all_tags = []
    all_ids = []

    preds = []
    ids = [d['id'] for d in data]
    with torch.no_grad():
        for i, d in enumerate(tqdm(test_loader)):
            out, _ = model(d["tokens"].to(args.device), None)
            # out = model(d["tokens"].to(args.device))
            _, pred = torch.max(out, 2)
            d['tags'] = d['tags'].to(args.device)
            for j, p in enumerate(pred):
                all_ids += ids[args.batch_size*i+j]
                all_pred += [list(map(lambda x:dataset.idx2label(x), list(filter(lambda x: (x != 9), p.cpu().tolist()))))]           
            all_tags = [i["tags"] for i in data]
            
    print(len(all_pred))
    print(len(all_tags))
    print('seqeval classification report')
    print(classification_report(all_tags, all_pred, mode='strict', scheme=IOB2))
    print("Joint Acc:", sum([1 if true == pred else 0 for true, pred in zip(all_tags, all_pred)]) / len(all_tags))
    print("Token Acc:", accuracy_score(all_tags, all_pred))


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--test_file",
        type=Path,
        help="Path to the test file.",
        default="./data/slot/eval.json"
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/slot/",
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        help="Path to model checkpoint.",
        required=True
    )

    # data
    parser.add_argument("--max_len", type=int, default=64)

    # model
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # data loader
    parser.add_argument("--batch_size", type=int, default=32)

    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda"
    )
    parser.add_argument('--seed', default=1, type=int, help="seed for model training")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    print(args)
    main(args)
