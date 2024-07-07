#!/usr/bin/env python

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import random
import argparse
import sys

from bsrnn import BSRNN
from bsrnn import Discriminator
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

    model = Discriminator().to("cuda")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    #optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=8, factor=0.8)
    l1loss = torch.nn.L1Loss(reduction='mean')

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

    train_loader = DataLoader(train, batch_size=1, shuffle=True, num_workers=2)
    val_loader = DataLoader(val, batch_size=1, shuffle=True, num_workers=2)

    epochI = 0
    best = 0
    while True:
        print("Epoch", epochI)
        batchI = 0

        epochLoss = 0.0
        for sample in tqdm(train_loader):
            waveform, waveform_speech = sample
            waveform = waveform.squeeze(0)
            waveform_speech = waveform_speech.squeeze(0)

            scale = random.uniform(0.02, 1.0)
            waveform *= scale
            waveform_speech *= scale
            stft_window = torch.hann_window(4096).to("cuda")

            waveform = torch.stft(waveform, n_fft=4096, hop_length=1024, return_complex=True, window=stft_window)
            waveform = torch.cat((waveform.real, waveform.imag), dim=1)

            waveform_speech = torch.stft(waveform_speech, n_fft=4096, hop_length=1024, return_complex=True, window=stft_window)
            waveform_speech = torch.cat((waveform_speech.real, waveform_speech.imag), dim=1)

            score_speech = model(waveform_speech)
            score_orig = model(waveform)

            # Maximize score_orig, minimize score_speech
            loss = score_speech + (1 - score_orig)

            loss.backward()
            epochLoss += loss.item()
            batchI += 1

            if args.batch_size == 1:
                optimizer.step()
                optimizer.zero_grad()
            else:
                if batchI % args.batch_size == 0:
                    optimizer.step()
                    optimizer.zero_grad()

        optimizer.step()
        optimizer.zero_grad()

        print("Epoch", epochI, "Loss", epochLoss / batchI, "lr", scheduler.get_last_lr())
        scheduler.step(epochLoss / batchI)
        epochI += 1

        with torch.no_grad():
            model.eval()
            valLoss = 0.0
            for sample in val_loader:

                waveform, waveform_speech = sample
                waveform = waveform.squeeze(0)
                waveform_speech = waveform_speech.squeeze(0)

                scale = random.uniform(0.02, 1.0)
                waveform *= scale
                waveform_speech *= scale
                stft_window = torch.hann_window(4096).to("cuda")

                waveform = torch.stft(waveform, n_fft=4096, hop_length=1024, return_complex=True, window=stft_window)
                waveform = torch.cat((waveform.real, waveform.imag), dim=1)

                waveform_speech = torch.stft(waveform_speech, n_fft=4096, hop_length=1024, return_complex=True, window=stft_window)
                waveform_speech = torch.cat((waveform_speech.real, waveform_speech.imag), dim=1)

                score_speech = model(waveform_speech)
                score_orig = model(waveform)
                loss = score_speech - score_orig

                valLoss += loss.item()

            print("Validation Loss", valLoss / len(val))
            if valLoss > best:
                best = valLoss
                torch.save(model.state_dict(), "discriminator.pth")

        model.train()


if __name__ == '__main__':
    main()
