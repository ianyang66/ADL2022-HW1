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
from utils import Vocab
from model_seqSlotClassifier import SeqSlotClassifier, SeqSlotLSTMClassifier

import time

TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]


def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    tag_idx_path = args.cache_dir / "tag2idx.json"
    tag2idx: Dict[str, int] = json.loads(tag_idx_path.read_text())

    torch.manual_seed(1)
    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
    datasets: Dict[str, SeqSlotDataset] = {
        split: SeqSlotDataset(split_data, vocab, tag2idx, args.max_len)
        for split, split_data in data.items()
    }
    # TODO: create DataLoader for train / dev datasets
    dataloaders: Dict[str, DataLoader] = {
        split: DataLoader(split_dataset, args.batch_size, shuffle=True, collate_fn=split_dataset.collate_fn) 
        for split, split_dataset in datasets.items()
    }

    # print(dataloaders[TRAIN]) <torch.utils.data.dataloader.DataLoader object at 0x000002665273D340>
    # print(dataloaders[DEV]) <torch.utils.data.dataloader.DataLoader object at 0x000002665273D5B0>

    embeddings = torch.load(args.cache_dir / "embeddings.pt")
    # TODO: init model and move model to target device(cpu / gpu)
    #model = SeqSlotClassifier(embeddings = embeddings, hidden_size = args.hidden_size, dropout = args.dropout, num_layers = args.num_layers, bidirectional = True, num_class = 10)
    model = SeqSlotLSTMClassifier(embeddings = embeddings, hidden_size = args.hidden_size, dropout = args.dropout, num_layers = args.num_layers, bidirectional = True, num_class = 10)
    device = args.device

    model.to(device)
    print(model)
    """
    try:
        ckpt = torch.load("./ckpt/slot/model.ckpt")
        model.load_state_dict(ckpt)
    except:
        print("Can't load model!")
    """

    # TODO: init optimizer
    #optimizer = torch.optim.RMSprop(model.parameters(), lr = args.lr, weight_decay = 0)
    #optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weigth_decay = 0)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay = 0)

    epoch_pbar = trange(args.num_epoch, desc="Epoch")
    num_class = datasets[TRAIN].num_classes
    criterion = nn.CrossEntropyLoss()
    #criterion = FocalLoss(num_class)
    best_acc = 0.0

    epoch_times = []
    for epoch in epoch_pbar:
        # TODO: Training loop - iterate over train dataloader and update model weights
        model.train()
        train_loss, train_acc, val_loss, val_acc = 0.0, 0.0, 0.0, 0.0
        start_time = time.perf_counter()
        tr = tqdm(dataloaders[TRAIN])
        for i, data in enumerate(tr):
            inputs, labels = data["tokens"], data["tags"]   
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            #out, _ = model(inputs, None)
            out = model(inputs)
            #print(labels.view(-1).shape)
            #print(out.view(-1,10).shape)
            loss = criterion(out.view(-1, 10), labels.view(-1))
            _, train_pred = torch.max(out, 2)
            loss.backward()
            optimizer.step()

            for j, label in enumerate(labels):
                train_acc += ((train_pred[j].cpu() == label.cpu()).sum().item() == args.max_len)
            train_loss += loss.item()
            tr.clear()

        # TODO: Evaluation loop - calculate accuracy and save model weights
        with torch.no_grad():
            #h = model.init_hidden(batch_size, device)
            model.eval()
            for i, dev_data in enumerate(tqdm(dataloaders[DEV])):
                inputs, labels = dev_data["tokens"].to(device), dev_data["tags"].to(device)   
                #out, _ = model(inputs, None)
                out = model(inputs)
                loss = criterion(out.view(-1, 10), labels.view(-1))
                _, val_pred = torch.max(out, 2)
                #print(val_pred)
                for j, label in enumerate(labels):
                    val_acc += ((val_pred[j].cpu() == label.cpu()).sum().item() == args.max_len)
                val_loss += loss.item()
            
            print(f"Epoch {epoch + 1}: Train Acc: {train_acc / len(dataloaders[TRAIN].dataset)}, Train Loss: {train_loss / len(dataloaders[TRAIN])}, Val Acc: {val_acc / len(dataloaders[DEV].dataset)}, Val Loss: {val_loss / len(dataloaders[DEV])}")
            ckp_dir = "./ckpt/slot/"
            if val_acc >= best_acc:
                best_acc = val_acc
                ckp_path = ckp_dir + '{}-model.ckpt'.format(epoch + 1)
                best_ckp_path = ckp_dir + 'best-model.ckpt'.format(epoch + 1)
                torch.save(model.state_dict(), ckp_path)
                torch.save(model.state_dict(), best_ckp_path)
                print(f"Save model with acc {val_acc / len(dataloaders[DEV].dataset)}")
        ckp_path = ckp_dir + '{}-model.ckpt'.format(epoch + 1)
        torch.save(model.state_dict(), ckp_path)
        print(f"Save model with acc {val_acc / len(dataloaders[DEV].dataset)}")
        current_time = time.perf_counter()
        epoch_times.append(current_time-start_time)    
        print("Total Time Elapsed: {} seconds".format(str(current_time-start_time)))
    print("Total Training Time: {} seconds".format(str(sum(epoch_times))))        
    # TODO: Inference on test set
    test_data = None
    with open("./data/slot/test.json", "r") as fp:
        test_data = json.load(fp)
    test_dataset = SeqSlotDataset(test_data, vocab, tag2idx, args.max_len)
    # TODO: create DataLoader for test dataset
    test_loader = torch.utils.data.DataLoader(test_dataset, shuffle = False, batch_size = args.batch_size, collate_fn = test_dataset.collate_fn)
    model.eval()
    #model.load_state_dict(torch.load("./ckpt/slot/model.ckpt"))
    ids = [d["id"] for d in test_data]
    # load weights into model

    # TODO: predict dataset
    preds = []
    ids = [d['id'] for d in test_data]
    with open("./pred.slot.csv", "w") as fp:
        fp.write("id,tags\n")
        with torch.no_grad():
            for i, d in enumerate(tqdm(test_loader)):
                #out, _ = model(d["tokens"].to(device), None)
                out = model(d["tokens"].to(device))
                _, pred = torch.max(out, 2)
                for j, p in enumerate(pred):
                    fp.write(f"{ids[args.batch_size*i+j]},{' '.join(list(map(lambda x:test_dataset.idx2label(x), list(filter(lambda x: (x != 9), p.tolist())))))}\n")

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/slot/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/slot/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/slot/",
    )

    # data
    parser.add_argument("--max_len", type=int, default=64)

    # model
    parser.add_argument("--hidden_size", type=int, default=2048)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-4)

    # data loader
    parser.add_argument("--batch_size", type=int, default=8)

    # training
    parser.add_argument(
            "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda:1"
    )
    parser.add_argument("--num_epoch", type=int, default=150)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)