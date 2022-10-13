import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import trange, tqdm
from dataset import SeqSlotDataset
from model_seqSlotClassifier import SeqSlotClassifier, SeqSlotLSTMClassifier
from utils import Vocab

import time
TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]

def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    tag_idx_path = args.cache_dir / "tag2idx.json"
    tag2idx: Dict[str, int] = json.loads(tag_idx_path.read_text())

    # print(args.test_file)
    # print(args.test_file.read_text())
    test_data = json.loads(args.test_file.read_text())
    test_dataset = SeqSlotDataset(test_data, vocab, tag2idx, args.max_len)
    
    # TODO: crecate DataLoader for test dataset
    test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False, collate_fn=test_dataset.collate_fn)
    embeddings = torch.load(args.cache_dir / "embeddings.pt")

    ckpt = torch.load(args.ckpt_path)
    # print(ckpt)
    model = SeqSlotClassifier(
        embeddings = embeddings,
        hidden_size = args.hidden_size,
        dropout = args.dropout,
        num_layers = args.num_layers,
        bidirectional = True,
        num_class = 10
    )
    """model = SeqSlotLSTMClassifier(
        embeddings = embeddings,
        hidden_size = args.hidden_size,
        dropout = args.dropout,
        num_layers = args.num_layers,
        bidirectional = True,
        num_class = 10
    )"""
    device = args.device
    # load weights into model
    model.load_state_dict(ckpt)
    model.eval()
    
    # TODO: predict dataset
    predictions = []
    ids = [data["id"] for data in test_data]
    with open(args.pred_file, "w") as f:
        f.write("id,tags\n")
        with torch.no_grad():
            for i, data in enumerate(tqdm(test_loader)):
                out, _ = model(data["tokens"].to(device), None)
                #out = model(data["tokens"].to(device))
                _, predictions = torch.max(out, 2)

                # TODO: write prediction to file (args.pred_file)
                for j, pred in enumerate(predictions):
                    f.write(f"{ids[args.batch_size*i+j]},{' '.join(list(map(lambda x:test_dataset.idx2label(x), list(filter(lambda x: (x != 9), pred.tolist())))))}\n")
        print("Write prediction file Done.")

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--test_file",
        type=Path,
        help="Path to the test file.",
        default="./data/slot/test.json",
        required=True
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
    parser.add_argument("--pred_file", type=Path, default="pred.tag.csv")

    # data
    parser.add_argument("--max_len", type=int, default=64)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # data loader
    parser.add_argument("--batch_size", type=int, default=32)

    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
