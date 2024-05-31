#!/usr/bin/env python

import torch
import torchaudio
from bsrnn import BSRNN
import os
import wandb

sample_bases = '/home/phh/d/d/d4/dnr_v2'


def samples(folder):
    s = [sample_bases + '/' + folder + '/' + x for x in os.listdir(sample_bases + '/' + folder)]
    # Filter out files
    s = [x for x in s if os.path.isdir(x)]
    s = [(x + '/mix.wav', x + '/speech.wav') for x in s]
    return s


def load_waveform(path):
    waveform, sample_rate = torchaudio.load(path)
    if waveform.shape[0] == 1:
        waveform = torch.cat((waveform, waveform), 0)
    return waveform.to("cuda")


class MyDataSet(torch.utils.data.Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return load_waveform(self.samples[idx][0]), load_waveform(self.samples[idx][1])


def main():
    wandb.init()
    model = BSRNN().to("cuda")
    wandb.watch(model)
    # List the folders  in $sample_bases / tr (every folder there is a train sample)

    optimizer = torch.optim.Adam(model.parameters(), lr=1.0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10)
    l1loss = torch.nn.L1Loss()

    train = samples('tr')
    train = MyDataSet(train)
    val = samples('cv')
    val = MyDataSet(val)
    test = samples('tt')
    test = MyDataSet(test)

    train_loader = torch.utils.data.DataLoader(train, batch_size=1, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val, batch_size=1, shuffle=True)

    epochI = 0
    while True:
        print("Epoch", epochI)
        batchI = 0

        epochLoss = 0.0
        sdr = 0.0
        for sample in train_loader:
            waveform, waveform_speech = sample
            waveform = waveform.squeeze(0)
            waveform_speech = waveform_speech.squeeze(0)
            # print(waveform.shape, waveform_speech.shape)
            # Forward the sample through the model
            x = model.forward(waveform)
            waveform_speech = waveform_speech[:, :x.shape[1]]
            loss = l1loss(x, waveform_speech)
            #print("x", x.mean().item(), x.max().item(), x.std().item(), x.min().item())
            #print("waveform_speech", waveform_speech.mean().item(), waveform_speech.max().item(),
            #      waveform_speech.std().item(), waveform_speech.min().item())
            # print(x.shape, "vs", waveform.shape)
            batchI += 1
            loss.backward()
            epochLoss += loss.item()
            sdr += 10 * torch.log10(torch.linalg.vector_norm(waveform_speech, ord=1).detach() / (x.shape[1] * loss.item() + 1e-9))

            if (batchI % 100) == 0:
                optimizer.step()
                optimizer.zero_grad()

        optimizer.step()
        optimizer.zero_grad()
        wandb.log({"train_loss": epochLoss / batchI, "train_sdr": sdr / batchI})

        print("Epoch", epochI, "Loss", epochLoss / batchI, "lr", scheduler.get_last_lr(), "sdr", sdr / batchI)
        scheduler.step(epochLoss / batchI)
        torch.save(model.state_dict(), "model.pth")
        epochI += 1

        with torch.no_grad():
            model.eval()
            valLoss = 0.0
            valSdr = 0.0
            for sample in val_loader:
                waveform, waveform_speech = sample
                waveform = waveform.squeeze(0)
                waveform_speech = waveform_speech.squeeze(0)

                # Forward the sample through the model
                x = model.forward(waveform)
                waveform_speech = waveform_speech[:, :x.shape[1]]
                loss = l1loss(x, waveform_speech)
                valLoss += loss.item()
                valSdr += 10 * torch.log10(torch.linalg.vector_norm(waveform_speech.item(), ord=1) / (loss.item() + 1e-9))
            print("Validation Loss", valLoss / len(val), "Validation SDR", valSdr / len(val))
            wandb.log({"val_loss": valLoss / len(val), "val_sdr": valSdr / len(val)})

        model.train()


if __name__ == '__main__':
    main()
