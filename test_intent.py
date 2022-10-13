import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import trange, tqdm
from dataset import SeqClsDataset
from model_seqCLSClassifier import SeqCLSClassifier, SeqCLSLSTMClassifier
from utils import Vocab

import time
TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]

def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())

    # print(args.test_file)
    # print(args.test_file.read_text())
    test_data = json.loads(args.test_file.read_text())
    test_dataset = SeqClsDataset(test_data, vocab, intent2idx, args.max_len)
    
    # TODO: crecate DataLoader for test dataset
    test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False, collate_fn=test_dataset.collate_fn)
    embeddings = torch.load(args.cache_dir / "embeddings.pt")

    ckpt = torch.load(args.ckpt_path)
    # print(ckpt)
    """model = SeqCLSLSTMClassifier(
        embeddings = embeddings,
        hidden_size = args.hidden_size,
        dropout = args.dropout,
        num_layers = args.num_layers,
        bidirectional = True,
        num_class = test_dataset.num_classes
    )"""
    model = SeqCLSLSTMClassifier(
        embeddings = embeddings,
        hidden_size = args.hidden_size,
        dropout = args.dropout,
        num_layers = args.num_layers,
        bidirectional = True,
        num_class = test_dataset.num_classes
    )
    
    device = args.device
    # load weights into model
    model.load_state_dict(ckpt)
    model.eval()
    val_acc = 0.0
    # TODO: predict dataset
    predictions = []
    ids = [data["id"] for data in test_data]
    with open(args.pred_file, "w") as f:
        f.write("id,intent\n")
        with torch.no_grad():
            for i, data in enumerate(tqdm(test_loader)):
                labels = data["intent"].to(device)
                labels = torch.autograd.Variable(labels).long()
                #out, _ = model(data["text"].to(device), None)
                out = model(data["text"].to(device))
                _, predictions = torch.max(out, 1)
                val_acc += (predictions.cpu() == labels.cpu()).sum().item()
                # TODO: write prediction to file (args.pred_file)
                for j, pred in enumerate(predictions):
                    f.write(f"{ids[args.batch_size*i+j]},{test_dataset.idx2label(pred.item())}\n")
        print("Write prediction file Done.")
        print(f"Val Acc: {val_acc / len(test_loader.dataset)}")
def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--test_file",
        type=Path,
        help="Path to the test file.",
        default="./data/intent/test.json",
        required=True
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/intent/",
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        help="Path to model checkpoint.",
        required=True
    )
    parser.add_argument("--pred_file", type=Path, default="pred_intent.csv")

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
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
