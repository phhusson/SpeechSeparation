#!/usr/bin/env python

import torch
from torch import nn
import torch.nn.functional as F


def pshape(*args):
    pass
    #print(*args)


band_features = 64
class NormRNNResidual(nn.Module):
    def __init__(self):
        super(NormRNNResidual, self).__init__()
        self.groupnorm = nn.GroupNorm(band_features, band_features)
        self.rnn = nn.LSTM(band_features, band_features, batch_first=True, num_layers=2)
        self.fc = nn.Linear(band_features, band_features)

    def forward(self, x: torch.Tensor):
        out = x
        out = out / out.norm()
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


class BSRNN(nn.Module):
    def __init__(self):
        super(BSRNN, self).__init__()
        # Take the STFT of each band, and output same sized-vector for each band
        self.bandFCs = nn.ModuleList([
            nn.Linear(8 * 2, band_features),
            nn.Linear(16 * 2, band_features),
            nn.Linear(32 * 2, band_features),
            nn.Linear(64 * 2, band_features),
            nn.Linear(128 * 2, band_features),
            nn.Linear(256 * 2, band_features),
            nn.Linear(512 * 2, band_features),
            nn.Linear((1024 + 9) * 2, band_features)
        ])

        num_lstm_layers = 0
        self.lstms = nn.Sequential()
        for j in range(num_lstm_layers):
            self.lstms.append(BandwiseLSTM())
            self.lstms.append(TimewiseLSTM())

        # Get back from the band features into full bands
        # Paper has hidden layer 512
        mask_estimation_mlp_hidden = 512
        self.bandFCs_back = nn.ModuleList([nn.Linear(band_features, mask_estimation_mlp_hidden) for _ in range(len(self.bandFCs))])
        self.bandFCs_back2 = nn.ModuleList([
            nn.Linear(mask_estimation_mlp_hidden, 8 * 2 * 2),
            nn.Linear(mask_estimation_mlp_hidden, 16 * 2 * 2),
            nn.Linear(mask_estimation_mlp_hidden, 32 * 2 * 2),
            nn.Linear(mask_estimation_mlp_hidden, 64 * 2 * 2),
            nn.Linear(mask_estimation_mlp_hidden, 128 * 2 * 2),
            nn.Linear(mask_estimation_mlp_hidden, 256 * 2 * 2),
            nn.Linear(mask_estimation_mlp_hidden, 512 * 2 * 2),
            nn.Linear(mask_estimation_mlp_hidden, (1024 + 9) * 2 * 2)
        ])
        self.bandFCs_back2_glu = nn.GLU()

    @staticmethod
    def generate_bandsplits():
        # Note: this splits in a logarithmic way, but maybe this makes the biggest bands too big
        return [
            8,
            16,
            32,
            64,
            128,
            256,
            512,
            1024 + 9
        ]


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
        # We want to split the frequencies in bands
        bandsplit = self.generate_bandsplits()
        current_band_start = 0
        bands = []
        for band in bandsplit:
            b = x[:, current_band_start:current_band_start + band, :]
            b /= b.norm()
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
            band = F.tanh(band)
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