#!/usr/bin/env python

import torch
from bsrnn import BSRNN
import wandb
import argparse

from m_dataset import samples, MyDataSet, train_infer
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser(description='Train a BSRNN model')
parser.add_argument('--datapath', type=str, default='/home/phh/d/d/d4/dnr_v2', help='Path to the dataset')
parser.add_argument('--mini', action='store_true', help='Use a small dataset')
args = parser.parse_args()

def main():
    wandb.init()
    model = BSRNN().to("cuda")
    wandb.watch(model)
    # List the folders  in $sample_bases / tr (every folder there is a train sample)

    optimizer = torch.optim.Adam(model.parameters(), lr=1.0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10)
    l1loss = torch.nn.L1Loss()

    train = samples(args.datapath, 'tr')
    if args.mini:
        train = train[:10]
    train = MyDataSet(train)
    val = samples(args.datapath, 'cv')
    if args.mini:
        val = val[:10]
    val = MyDataSet(val)
    test = samples(args.datapath, 'tt')
    test = MyDataSet(test)

    train_loader = DataLoader(train, batch_size=1, shuffle=True)
    val_loader = DataLoader(val, batch_size=1, shuffle=True)

    epochI = 0
    while True:
        print("Epoch", epochI)
        batchI = 0

        epochLoss = 0.0
        epochSdr = 0.0
        for sample in train_loader:
            loss, sdr = train_infer(model, sample, l1loss)

            batchI += 1
            sdr.backward()
            epochLoss += loss.item()
            epochSdr += sdr.item()

            if (batchI % 100) == 0:
                optimizer.step()
                optimizer.zero_grad()

        optimizer.step()
        optimizer.zero_grad()
        wandb.log({"train_loss": epochLoss / batchI, "train_sdr": epochSdr / batchI})

        print("Epoch", epochI, "Loss", epochLoss / batchI, "lr", scheduler.get_last_lr(), "sdr", epochSdr / batchI)
        scheduler.step(epochLoss / batchI)
        torch.save(model.state_dict(), "model.pth")
        epochI += 1

        with torch.no_grad():
            model.eval()
            valLoss = 0.0
            valSdr = 0.0
            for sample in val_loader:
                loss, sdr = train_infer(model, sample, l1loss)
                valLoss += loss.item()
                valSdr += sdr.item()

            print("Validation Loss", valLoss / len(val), "Validation SDR", valSdr / len(val))
            wandb.log({"val_loss": valLoss / len(val), "val_sdr": valSdr / len(val)})

        model.train()


if __name__ == '__main__':
    main()
