import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.notebook import trange, tqdm

from dataset import SeqClsDataset
from utils import Vocab
from model_seqCLSClassifier import SeqCLSClassifier, SeqCLSLSTMClassifier

import time

TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]

def clip_gradient(model, clip_value):
    params = list(filter(lambda p: p.grad is not None, model.parameters()))
    for p in params:
        p.grad.data.clamp_(-clip_value, clip_value)

def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())

    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
    datasets: Dict[str, SeqClsDataset] = {
        split: SeqClsDataset(split_data, vocab, intent2idx, args.max_len)
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
    # print(embeddings)
    
    # TODO: init model and move model to target device(cpu / gpu)
    # model = SeqCLSClassifier(embeddings, hidden_size = args.hidden_size, dropout = args.dropout, num_layers = args.num_layers, bidirectional = args.bidirectional, num_class = datasets[TRAIN].num_classes)
    model = SeqCLSLSTMClassifier(embeddings, hidden_size = args.hidden_size, dropout = args.dropout, num_layers = args.num_layers, bidirectional = args.bidirectional, num_class = datasets[TRAIN].num_classes)

    device = args.device
    model.to(device)
    print(model)

    """try:
        ckpt = torch.load("./ckpt/intent/model.ckpt")
        model.load_state_dict(ckpt)
    except:
        print("Can't load model!")
    """
    
    batch_size = args.batch_size
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

        h = model.init_hidden(batch_size, device)
        for i, data in enumerate(tqdm(dataloaders[TRAIN])):
            inputs, labels = data["text"].to(device), data["intent"].to(device)
            labels = torch.autograd.Variable(labels).long()
            optimizer.zero_grad()
            out = model(inputs) #lstm
            #out,_ = model(inputs, None)#gru
            #print(labels.shape)
            loss = criterion(out, labels)
            
            _, train_pred = torch.max(out, 1)
            loss.backward()
            optimizer.step()
            train_acc += (train_pred.cpu() == labels.cpu()).sum().item()
            train_loss += loss.item()
 
        # TODO: Evaluation loop - calculate accuracy and save model weights
        with torch.no_grad():
            #h = model.init_hidden(batch_size, device)
            model.eval()
            for i, dev_data in enumerate(tqdm(dataloaders[DEV])):
                inputs, labels = dev_data["text"].to(device), dev_data["intent"].to(device)
                labels = torch.autograd.Variable(labels).long()
                out = model(inputs) #lstm
                #out,_ = model(inputs, None) #gru
                #print(out.size())
                #print(labels.size())
                loss = criterion(out, labels)
                _, val_pred = torch.max(out, 1)
                val_acc += (val_pred.cpu() == labels.cpu()).sum().item()
                val_loss += loss.item()
            
            print(f"Epoch {epoch + 1}: Train Acc: {train_acc / len(dataloaders[TRAIN].dataset)}, Train Loss: {train_loss / len(dataloaders[TRAIN])}, Val Acc: {val_acc / len(dataloaders[DEV].dataset)}, Val Loss: {val_loss / len(dataloaders[DEV])}")
            ckp_dir = "./ckpt/intent/"
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
    with open("./data/intent/test.json", "r") as fp:
        test_data = json.load(fp)
    test_dataset = SeqClsDataset(test_data, vocab, intent2idx, args.max_len)
    # Create DataLoader for test dataset
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=test_dataset.collate_fn)
    model.eval()
    
    predictions = []
    ids = [data["id"] for data in test_data]
    with open("./pred_intent.csv", "w") as f:
        f.write("id,intent\n")
        with torch.no_grad():
            for i, data in enumerate(tqdm(test_loader)):
                #out, _ = model(data["text"].to(device), None) #gru
                out = model(data["text"].to(device)) #lstm
                _, predictions = torch.max(out, 1)
                for j, pred in enumerate(predictions):
                    f.write(f"{ids[batch_size*i+j]},{test_dataset.idx2label(pred.item())}\n")
        print("Write prediction file Done.")            

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/intent/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/intent/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/intent/",
    )

    # data
    parser.add_argument("--max_len", type=int, default=64)

    # model
    parser.add_argument("--hidden_size", type=int, default=4096)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-3)

    # data loader
    parser.add_argument("--batch_size", type=int, default=32)

    # training
    parser.add_argument(
            "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda:0"
    )
    parser.add_argument("--num_epoch", type=int, default=100)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)