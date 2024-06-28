#!/usr/bin/env python

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import wandb
import argparse
import sys

from bsrnn import BSRNN
from m_dataset import samples, MyDataSet, train_infer

parser = argparse.ArgumentParser(description='Train a BSRNN model')
parser.add_argument('--datapath', type=str, default='/home/phh/d/d/d4/dnr_v2', help='Path to the dataset')
parser.add_argument('--mini', action='store_true', help='Use a small dataset')
parser.add_argument('--dump_graph', action='store_true', help='Dump the compute graph to graph.dot')
parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
parser.add_argument('--resume', action='store_true', help='Reload model')
parser.add_argument('--loss_sdr', action='store_true', help='Use SDR as loss')
args = parser.parse_args()

def main():
    torch.multiprocessing.set_start_method('spawn')

    model = BSRNN().to("cuda")
    if args.dump_graph:
        from torchview import draw_graph
        model = model.to('meta')
        x = torch.randn(2, 48000*60)
        x = torch.stft(x, n_fft=4096, hop_length=512, return_complex=True)
        model_graph = draw_graph(model, input_data=x, device='meta')
        with open('graph.dot', 'w') as f:
            f.write(model_graph.visual_graph.source)

        sys.exit(0)

    wandb.init()
    wandb.watch(model)
    if args.resume:
        model.load_state_dict(torch.load("model.pth"))

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)
    l1loss = torch.nn.L1Loss(reduction='mean')

    train = samples(args.datapath, 'tr', with_s2s = True)
    if args.mini:
        train = train[:10]
    train = MyDataSet(train)
    val = samples(args.datapath, 'cv')
    if args.mini:
        val = val[:10]
    val = MyDataSet(val)
    test = samples(args.datapath, 'tt')
    test = MyDataSet(test)

    train_loader = DataLoader(train, batch_size=1, shuffle=True, num_workers=2)
    val_loader = DataLoader(val, batch_size=1, shuffle=True, num_workers=2)

    epochI = 0
    while True:
        print("Epoch", epochI)
        batchI = 0

        epochLoss = 0.0
        epochSdr = 0.0
        for sample in tqdm(train_loader):
            loss, sdr = train_infer(model, sample, l1loss)

            batchI += 1
            if args.loss_sdr:
                toBackward = -sdr
            else:
                toBackward = loss
            toBackward.backward()
            epochLoss += loss.item()
            epochSdr += sdr.item()

            if args.batch_size == 1:
                optimizer.step()
                optimizer.zero_grad()
            else:
                if batchI % args.batch_size == 0:
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
