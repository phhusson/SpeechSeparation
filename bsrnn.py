#!/usr/bin/env python

import torch
from torch import nn
import torch.nn.functional as F


def pshape(*args):
    pass
    #print(*args)


# Takes as input [C; A; B] and outputs [C; A; B]; where C are ignored, A is
band_features = 64
class NormRNNResidual(nn.Module):
    def __init__(self):
        super(NormRNNResidual, self).__init__()
        self.groupnorm = nn.InstanceNorm1d(band_features)
        self.rnn = nn.LSTM(band_features, band_features, batch_first=True, num_layers=2)
        self.fc = nn.Linear(band_features, band_features)

    def forward(self, x: torch.Tensor):
        out = x
        out = self.groupnorm(out.permute((0, 2, 1))).permute((0, 2, 1))
        out = self.rnn(out)[0]
        out = self.fc(out)
        out = out + x
        return out

# Take as input [2; T; nBands; 128] and output [2; T; nBands; 128]
class TimewiseLSTM(nn.Module):
    def __init__(self):
        super(TimewiseLSTM, self).__init__()
        self.m = NormRNNResidual()

    def forward(self,x: torch.Tensor):
        # X is [2; T; nBands; 128], we need [2 * nBands ; T; 128]
        # First permute to [2; nBands; T; 128]
        x = x.permute((0, 2, 1, 3))
        # Then reshape to [2 * nBands; T; 128]
        x = x.reshape((2 * x.shape[1], x.shape[2], x.shape[3]))

        out = self.m(x)
        # Reshape back to [2; nBands; T; 128]
        out = out.reshape((2, -1, x.shape[1], x.shape[2]))
        # Permute back to [2; T; nBands; 128]
        out = out.permute((0, 2, 1, 3))

        return out

# Take as input [2; T; nBands; 128] and output [2; T; nBands; 128]
class BandwiseLSTM(nn.Module):
    def __init__(self):
        super(BandwiseLSTM, self).__init__()
        self.m = NormRNNResidual()

    def forward(self, x: torch.Tensor):
        # X is [2; T; nBands; 128], we need [2 * T ; nBands; 128]
        x = x.reshape((x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))

        out = self.m(x)
        # out is [2 * T; nBands; 128]
        # Reshape back to [2; T; nBands; 128]
        out = out.reshape((2, -1, x.shape[1], x.shape[2]))
        return out

class BandwiseFC(nn.Module):
    def __init__(self):
        super(BandwiseFC, self).__init__()
        nBands = len(generate_bandsplits())
        self.fc1 = nn.Linear(band_features * nBands, band_features * nBands)
        self.fc2 = nn.Linear(band_features * nBands, band_features * nBands)

    def forward(self, x: torch.Tensor):
        # X is [2; T; nBands; 128], we need [2 ; T ; nBands * 128]
        out = x.reshape((x.shape[0], x.shape[1], -1))
        out = self.fc1(out)
        out = F.tanh(out)
        out = self.fc2(out)
        out = F.tanh(out)
        # Reshape back to [2; T; nBands; 128]
        out = out.reshape((2, x.shape[1], len(generate_bandsplits()), band_features))
        return out

def generate_bandsplits():
    # Note: this splits in a logarithmic way, but maybe this makes the biggest bands too big
    v = [
        (10, 0),
        (10, 10),
        (10, 20),
        (10, 30),
        (10, 40),
        (10, 50),
        (20, 60),
        (20, 80),
        (50, 100),
        (50, 150),
        (50, 200),
        (50, 250),
        (100, 300),
        (100, 400),
        (250, 500),
        (250, 750),
        (500, 1000),
    ]
    v = [v[0] for x in v]
    return v + [2049 - sum(v)]

class BSRNN(nn.Module):
    def __init__(self):
        super(BSRNN, self).__init__()
        # Take the STFT of each band, and output same sized-vector for each band
        self.bandFCs = nn.ModuleList([
            nn.Linear(x * 2, band_features) for x in generate_bandsplits()
        ])

        num_lstm_layers = 2
        self.lstms = nn.Sequential()
        for j in range(num_lstm_layers):
            self.lstms.append(BandwiseFC())
            self.lstms.append(TimewiseLSTM())

        # Get back from the band features into full bands
        # Paper has hidden layer 512
        mask_estimation_mlp_hidden = 512
        self.bandFCs_back = nn.ModuleList([
            nn.Linear(band_features, mask_estimation_mlp_hidden) for _ in range(len(self.bandFCs))
        ])
        self.bandFCs_back_prelu = nn.ModuleList([nn.PReLU() for _ in range(len(self.bandFCs))])
        self.bandFCs_back2 = nn.ModuleList([
            nn.Linear(mask_estimation_mlp_hidden, x * 2 * 2) for x in generate_bandsplits()
        ])
        self.bandFCs_back2_glu = nn.GLU()

    def forward(self, x):
        # Signal is 48kHz
        # x is [2; T] where T is the number of samples
        # Do STFTs on the input, with a 4096 Window, and 1024 hop length (so 75% overlap)
        # (So one sample will be seen 4 times)
        og_mean = x.mean()
        og_std = x.std()
        x = (x - og_mean) / og_std

        pshape("Input", x.shape)
        x = torch.stft(x, n_fft=4096, hop_length=512, return_complex=True, window=torch.hann_window(4096).to("cuda"))

        pshape("STFT", x.shape)
        # x is now [2; F; T/512] where F is the number of frequencies, here 2049
        # From here on, we stop saying "T/512" but just T
        # We want to split the frequencies in bands
        bandsplit = generate_bandsplits()
        current_band_start = 0
        bands = []
        for band in bandsplit:
            b = x[:, current_band_start:current_band_start + band, :]
            b /= torch.linalg.vector_norm(b)
            bands.append(b)
            current_band_start += band

        # Now we have the bands, we can do the band specific processing
        band_outputs = []
        for i, band in enumerate(bands):
            # Permute the band to [2; T; F]
            band = band.permute((0, 2, 1))
            # Concatenate the real and imaginary parts of band
            band = torch.cat((band.real, band.imag), 2)
            pshape("Band ", i, band.shape)
            y = self.bandFCs[i](band)
            band_outputs.append(y)
            pshape("Band output", i, y.shape)

        # band_outputs is python array of 3D tensors, make it a 4D tensor
        band_outputs = torch.stack(band_outputs, 2)
        bands_with_time_and_bands = self.lstms(band_outputs)

        # Now we have the bands with time and bands, we can do the band specific processing
        mask_estimations = []
        for i, ogBandFc in enumerate(self.bandFCs):
            band = bands_with_time_and_bands[:, :, i, :]
            band = self.bandFCs_back[i](band)
            band = self.bandFCs_back_prelu[i](band)
            band = self.bandFCs_back2[i](band)
            band = self.bandFCs_back2_glu(band)
            pshape("Band back pre", i, band.shape)
            # band is [2; T; <filter size * 2>], we want to split it back into real and imaginary parts
            band = band.reshape((2, -1, 2, band.shape[2] // 2))
            band = torch.complex(band[:, :, 0, :], band[:, :, 1, :])
            pshape("Band back", i, band.shape)
            mask_estimations.append(band)
        mask_estimations = torch.cat(mask_estimations, 2)

        mask_estimations = mask_estimations.permute((0, 2, 1))

        pshape("Final mask is", mask_estimations.shape)
        pshape("x is ", x.shape)

        x = x * mask_estimations
        x = torch.istft(x, n_fft=4096, hop_length=512, window=torch.hann_window(4096).to("cuda"))
        x = x * og_std + og_mean

        return x
