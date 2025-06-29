#!/usr/bin/env python

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import wandb
import argparse
import sys

from bsrnn import BSRNN
from bsrnn import Discriminator
from m_dataset import samples, MyDataSet, train_infer, mix_samples, MixDataSet

parser = argparse.ArgumentParser(description='Train a BSRNN model')
parser.add_argument('--datapath', type=str, default='/nvme1/ML/DnR/dnr_v2', help='Path to the dataset')
parser.add_argument('--mini', action='store_true', help='Use a small dataset')
parser.add_argument('--dump_graph', action='store_true', help='Dump the compute graph to graph.dot')
parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
parser.add_argument('--resume', action='store_true', help='Reload model')
parser.add_argument('--loss_sdr', action='store_true', help='Use SDR as loss')
args = parser.parse_args()

batch_size = 1
if args.batch_size:
    batch_size = args.batch_size
batch_size_f = batch_size

def main():
    global batch_size, batch_size_f
    torch.multiprocessing.set_start_method('spawn')

    model = BSRNN().to("cuda")
    discriminator = None
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

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay = 0.01)
    #optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.8)
    l1loss = torch.nn.L1Loss(reduction='mean')

    # Train is dnr dataset
    if False:
        train = samples(args.datapath, 'tr', with_s2s = False, with_rnnoise = True)
        if args.mini:
            train = train[:10]
        train = MyDataSet(train, train = True)
    # Train is remixed samples, of dnr + other sources
    else:
        trains = [
                (args.datapath + '/tr/', 1),
                ('/nvme1/ML/DnR/phh_dnr/twitch-lownoise', 10),
                ('/nvme1/ML/DnR/VocalSound/phh', 1)
        ]
        trains = [(mix_samples(x[0]), x[1]) for x in trains]
        train = (sum([x[0][0] * x[1] for x in trains], []), sum([x[0][1] * x[1] for x in trains], []))
        train = MixDataSet(train)

    val = samples(args.datapath, 'cv')
    val2 = samples(args.datapath, 'cv', with_rnnoise = True)
    if args.mini:
        val = val[:10]
        val2 = val2[:10]
    val = MyDataSet(val)
    val2 = MyDataSet(val2)
    test = samples(args.datapath, 'tt')
    test = MyDataSet(test)

    train_loader = DataLoader(train, batch_size=1, shuffle=True, num_workers=8)
    val_loader = DataLoader(val, batch_size=1, shuffle=True, num_workers=2)
    val2_loader = DataLoader(val2, batch_size=1, shuffle=True, num_workers=2)

    epochI = 0
    best = 1.0e30
    bestTrainLoss = 1.0e30
    while True:
        print("Epoch", epochI)
        batchI = 0

        epochLoss = 0.0
        epochSdr = 0.0
        for sample in tqdm(train_loader):
            loss, sdr, sdr2, sdr3 = train_infer(model, discriminator, sample, l1loss)

            batchI += 1
            if args.loss_sdr:
                toBackward = -sdr
            else:
                toBackward = loss
            toBackward.backward()
            epochLoss += loss.item()
            epochSdr += sdr.item()

            if batch_size == 1:
                optimizer.step()
                optimizer.zero_grad()
            else:
                if batchI % batch_size == 0:
                    optimizer.step()
                    optimizer.zero_grad()
        if epochLoss > bestTrainLoss:
            batch_size_f *= 1.02
            if batch_size_f > batchI:
                batch_size_f = batchI
            batch_size = int(batch_size_f)
            print("Setting batch size to", batch_size)
        else:
            bestTrainLoss = epochLoss

        optimizer.step()
        optimizer.zero_grad()

        print("Epoch", epochI, "Loss", epochLoss / batchI, "lr", scheduler.get_last_lr(), "sdr", epochSdr / batchI)
        #scheduler.step(epochLoss / batchI)
        epochI += 1

        with torch.no_grad():
            model.eval()
            valLoss = 0.0
            val2Loss = 0.0
            valSdr = 0.0
            valSdr2 = 0.0
            valSiSdr = 0.0
            for sample in val_loader:
                loss, sdr, sdr2, sisdr = train_infer(model, None, sample, l1loss)
                valLoss += loss.item()
                valSdr += sdr.item()
                valSdr2 += sdr2.item()
                valSiSdr += sisdr.item()

            val2Sdr = 0.0
            val2SiSdr = 0.0
            for sample in val2_loader:
                loss, sdr, sdr2, sisdr = train_infer(model, None, sample, l1loss)
                val2Sdr += sdr.item()
                val2SiSdr += sisdr.item()
                val2Loss += loss.item()

            print("Validation Loss", valLoss / len(val), "Validation SDR", valSdr / len(val))
            wandb.log({
                "val_loss": valLoss / len(val),
                "val_sdr": valSdr / len(val),
                "val_sisdr": valSiSdr / len(val),
                "val_rnnoise_sdr": val2Sdr / len(val),
                "val_rnnoise_sisdr": val2SiSdr / len(val),
                "train_loss": epochLoss / batchI,
                "train_sdr": epochSdr / batchI,
                "batch_size_f": batch_size_f,
                "lr": scheduler.get_last_lr()[0],
            })
            if val2Loss < best:
                print("...Saving")
                best = val2Loss
                torch.save(model.state_dict(), "model.pth")
                torch.save(optimizer.state_dict(), "optimizer.pth")
            torch.save(model.state_dict(), "model-always.pth")
            torch.save(optimizer.state_dict(), "optimizer-always.pth")

        model.train()


if __name__ == '__main__':
    main()
