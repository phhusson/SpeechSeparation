#!/usr/bin/env python

import torch
from torch import nn
import torch.nn.functional as F


def pshape(*args):
    pass
    #print(*args)


band_features = 64
# Takes as input [C; A; B] and outputs [C; A; B]; where C are ignored, A is
class NormRNNResidual(nn.Module):
    def __init__(self, bidirectional = False):
        super(NormRNNResidual, self).__init__()
        self.groupnorm = nn.InstanceNorm1d(band_features, track_running_stats=True)
        self.fc_in = nn.Linear(band_features, band_features)
        self.rnn = nn.LSTM(band_features, band_features, batch_first=True, num_layers=2, bidirectional=bidirectional)
        n = band_features
        if bidirectional:
            n = band_features * 2
        self.fc = nn.Linear(n, band_features)

    def forward(self, x: torch.Tensor):
        out = x
        #out = self.groupnorm(out.permute((0, 2, 1))).permute((0, 2, 1))
        out = self.fc_in(out)
        out = self.rnn(out)[0]
        out = self.fc(out)
        out = out + x
        return out

    def forward_recurrent(self, x, state):
        out = x
        #out = self.groupnorm(out.permute((0, 2, 1))).permute((0, 2, 1))
        out = self.fc_in(out)
        out = self.rnn(out, state)[0]
        out = self.fc(out)
        out = out + x
        return out, state

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

    def forward_recurrent(self, x, state):
        # X is [2; nBands; 128], we need [2 * nBands; 1; 128]
        x = x.reshape( (-1, 1, band_features) )
        x, state = self.m.forward_recurrent(x, (state[0], state[1]))
        x = x.reshape( (2, -1, band_features) )
        return x, torch.stack(state, dim=0)

# Take as input [2; T; nBands; 128] and output [2; T; nBands; 128]
class BandwiseLSTM(nn.Module):
    def __init__(self):
        super(BandwiseLSTM, self).__init__()
        self.m = NormRNNResidual(bidirectional = True)

    def forward(self, x: torch.Tensor):
        # X is [2; T; nBands; 128], we need [2 * T ; nBands; 128]
        x = x.reshape((x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))

        out = self.m(x)
        # out is [2 * T; nBands; 128]
        # Reshape back to [2; T; nBands; 128]
        out = out.reshape((2, -1, x.shape[1], x.shape[2]))
        return out

    def forward_recurrent(self, x):
        return self.m(x)

class BandwiseFC(nn.Module):
    def __init__(self):
        super(BandwiseFC, self).__init__()
        nBands = len(generate_bandsplits())
        self.fc1 = nn.Linear(band_features * nBands, band_features * nBands)
        self.fc2 = nn.Linear(band_features * nBands, band_features * nBands)
        self.fc3 = nn.Linear(band_features * nBands, band_features * nBands)

    def forward(self, x: torch.Tensor):
        # X is [2; T; nBands; 128], we need [2 ; T ; nBands * 128]
        out = x.reshape((x.shape[0], x.shape[1], -1))
        out = self.fc1(out)
        out = F.tanh(out)
        out = self.fc2(out)
        out = F.tanh(out)
        out = self.fc3(out)
        # Reshape back to [2; T; nBands; 128]
        out = out.reshape((2, x.shape[1], len(generate_bandsplits()), band_features))
        return out

def generate_bandsplits():
    #v = [
    #    (4, 0),
    #    (4, 4),
    #    (4, 8),
    #    (4, 12),
    #    (8, 16),
    #    (8, 24),
    #    (16, 32),
    #    (16, 48),
    #    (64, 64),
    #    (128, 128),
    #    (256, 256),
    #    (512, 512),
    #]
    v = [
        (5, 0),
        (5, 5),
        (5, 10),
        (5, 15),
        (10, 20),
        (10, 30),
        (10, 40),
        (10, 50),
        (10, 60),
        (10, 70),
        (10, 80),
        (10, 90),
        (50, 100),
        (50, 150),
        (50, 200),
        (50, 250),
        (50, 300),
        (50, 350),
        (50, 400),
        (50, 450),
        (100, 500),
        (100, 600),
        (100, 700),
        (100, 800),
        (100, 900),
        (250, 1000),
        (250, 1250),
        (250, 1500),
    ]
    pos = 0
    for x,y in v:
        assert y == pos
        pos += x
    v = [x[0] for x in v]
    return v + [2049 - sum(v)]

class BSRNN(nn.Module):
    def __init__(self):
        super(BSRNN, self).__init__()
        # Take the STFT of each band, and output same sized-vector for each band
        self.bandFCs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(x * 2, band_features),
                nn.LeakyReLU(),
                nn.Linear(band_features, band_features),
            ) for x in generate_bandsplits()
        ])

        num_lstm_layers = 4
        self.lstms = nn.Sequential()
        for j in range(num_lstm_layers):
            self.lstms.append(BandwiseLSTM())
            self.lstms.append(TimewiseLSTM())

        # Get back from the band features into full bands
        # Paper has hidden layer 512
        mask_estimation_mlp_hidden = band_features * 2 * 2
        self.bandFCs_back = nn.ModuleList([
            nn.Sequential(
                nn.Linear(band_features, mask_estimation_mlp_hidden),
                nn.LeakyReLU(),
                nn.Linear(mask_estimation_mlp_hidden, x * 2 * 2),
                nn.GLU(),
            ) for x in generate_bandsplits()])

    # Signal is 48kHz
    # pre-x is [2; T] where T is the number of samples
    # Caller do STFTs on the input, with a 4096 Window, and 512 hop length (so 88% overlap)
    # (So one sample will be seen 8 times)
    # Input `x` of this function is [2; F; T/512] where F is the number of frequencies, here 2049
    # From here on, we stop saying "T/512" but just T
    def forward(self, x):

        pshape("STFT", x.shape)
        bandsplit = generate_bandsplits()
        current_band_start = 0
        bands = []
        # We want to split the frequencies in bands
        for band in bandsplit:
            b = x[:, current_band_start:current_band_start + band, :]
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

        return x

    def forward_recurrent(self, x, state):
        pshape("STFT", x.shape)
        bandsplit = generate_bandsplits()
        current_band_start = 0
        bands = []
        # We want to split the frequencies in bands
        for band in bandsplit:
            b = x[:, current_band_start:current_band_start + band]
            bands.append(b)
            current_band_start += band

        # Now we have the bands, we can do the band specific processing
        band_outputs = []
        for i, band in enumerate(bands):
            band = torch.cat((band.real, band.imag), 1)
            pshape("Band ", i, band.shape)
            y = self.bandFCs[i](band)
            band_outputs.append(y)
            pshape("Band output", i, y.shape)

        # band_outputs is python array of 2D tensors, make it a 3D tensor
        band_outputs = torch.stack(band_outputs, 1)
        state_i = 0
        new_state = []
        for layer in self.lstms:
            pshape("Layer ", type(layer))
            if type(layer) is TimewiseLSTM:
                band_outputs, s = layer.forward_recurrent(band_outputs, state[state_i])
                state_i += 1
                new_state += [s]
            else:
                band_outputs = layer.forward_recurrent(band_outputs)
            pshape(band_outputs.shape)
        new_state = torch.stack(new_state, dim=0)


        bands_with_time_and_bands = band_outputs

        # Now we have the bands with time and bands, we can do the band specific processing
        mask_estimations = []
        for i, ogBandFc in enumerate(self.bandFCs):
            band = bands_with_time_and_bands[:, i, :]
            band = self.bandFCs_back[i](band)
            pshape("Band back pre", i, band.shape)
            # band is [2; T; <filter size * 2>], we want to split it back into real and imaginary parts
            band = band.reshape((2,  2, band.shape[1] // 2))
            band = torch.complex(band[:, 0, :], band[:, 1, :])
            pshape("Band back", i, band.shape)
            mask_estimations.append(band)
        mask_estimations = torch.cat(mask_estimations, 1)

        pshape("Final mask is", mask_estimations.shape)
        pshape("x is ", x.shape)

        x = x * mask_estimations

        return x, new_state
