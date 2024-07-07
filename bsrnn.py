#!/usr/bin/env python

import torch
from torch import nn
import torch.nn.functional as F
import math

def pshape(*args):
    pass
    #print(*args)

# This is a module to do an InstanceNorm1d on the last dim of the tensor, instead of the middle one
class XXCInstanceNorm1d(nn.Module):
    def __init__(self, num_channels):
        super(XXCInstanceNorm1d, self).__init__()
        self.groupnorm = nn.InstanceNorm1d(num_channels, track_running_stats=True)

    def forward(self, x):
        out = x.reshape( (-1, x.shape[-2], x.shape[-1]) )
        out = out.permute( (0, 2, 1) )
        out = self.groupnorm(out)
        out = out.permute( (0, 2, 1) )
        out = out.reshape(x.shape)
        return out

band_features = 64
merge_channels = True
# Takes as input [C; A; B] and outputs [C; A; B]; where C are ignored, A is
class NormRNNResidual(nn.Module):
    def __init__(self, bidirectional = False, with_groupnorm = None):
        super(NormRNNResidual, self).__init__()
        self.groupnorm = None
        if with_groupnorm:
            self.groupnorm = nn.InstanceNorm1d(with_groupnorm, track_running_stats=True)
        self.fc_in = nn.Linear(band_features, band_features)
        self.rnn = nn.LSTM(band_features, band_features, batch_first=True, num_layers=2, bidirectional=bidirectional)
        n = band_features
        if bidirectional:
            n = band_features * 2
        self.fc = nn.Linear(n, band_features)

    def forward(self, x: torch.Tensor):
        out = x
        if self.groupnorm:
            out = self.groupnorm(out.reshape((x.shape[0], -1)).permute( (1, 0))).permute( (1, 0)).reshape(x.shape)
        out = self.fc_in(out)
        out = self.rnn(out)[0]
        out = self.fc(out)
        out = out + x
        return out

    def forward_recurrent(self, x, state):
        out = x
        if self.groupnorm:
            out = self.groupnorm(out.reshape((x.shape[0], -1))).reshape(x.shape)
        out = self.fc_in(out)
        out, state = self.rnn(out, state)
        out = self.fc(out)
        out = out + x
        return out, state

# Take as input [2; T; nBands; 128] and output [2; T; nBands; 128]
class TimewiseLSTM(nn.Module):
    def __init__(self):
        super(TimewiseLSTM, self).__init__()
        self.m = NormRNNResidual()

    def forward(self,x: torch.Tensor):
        nChannels = x.shape[0]
        # X is [C; T; nBands; 128], we need [2 * nBands ; T; 128]
        # First permute to [2; nBands; T; 128]
        x = x.permute((0, 2, 1, 3))
        # Then reshape to [2 * nBands; T; 128]
        x = x.reshape((nChannels * x.shape[1], x.shape[2], x.shape[3]))

        out = self.m(x)
        # Reshape back to [2; nBands; T; 128]
        out = out.reshape((nChannels, -1, x.shape[1], x.shape[2]))
        # Permute back to [2; T; nBands; 128]
        out = out.permute((0, 2, 1, 3))

        return out

    def forward_recurrent(self, x, state):
        # X is [2; nBands; 128], we need [2 * nBands; 1; 128]
        nChannels = x.shape[0]
        x = x.reshape( (-1, 1, band_features) )
        x, state = self.m.forward_recurrent(x, (state[0], state[1]))
        x = x.reshape( (nChannels, -1, band_features) )
        return x, torch.stack(state, dim=0)

# Take as input [2; T; nBands; 128] and output [2; T; nBands; 128]
class BandwiseLSTM(nn.Module):
    def __init__(self):
        super(BandwiseLSTM, self).__init__()
        nBands = len(generate_bandsplits())
        self.m = NormRNNResidual(bidirectional = True)

    def forward(self, x: torch.Tensor):
        # X is [2; T; nBands; 128], we need [2 * T ; nBands; 128]
        nChannels = x.shape[0]
        x = x.reshape((nChannels * x.shape[1], x.shape[2], x.shape[3]))

        out = self.m(x)
        # out is [2 * T; nBands; 128]
        # Reshape back to [2; T; nBands; 128]
        out = out.reshape((nChannels, -1, x.shape[1], x.shape[2]))
        return out

    def forward_recurrent(self, x):
        return self.m(x)

class BandwiseFC(nn.Module):
    def __init__(self):
        super(BandwiseFC, self).__init__()
        nBands = len(generate_bandsplits())
        self.layers = nn.Sequential(
                XXCInstanceNorm1d(band_features * nBands),
                nn.Linear(band_features * nBands, band_features * nBands),
                nn.Sigmoid(),
                nn.Linear(band_features * nBands, band_features * nBands),
                nn.Sigmoid(),
                nn.Linear(band_features * nBands, band_features * nBands),
                nn.Sigmoid(),
                nn.Linear(band_features * nBands, band_features * nBands),
        )

    def forward(self, x: torch.Tensor):
        # X is [2; T; nBands; 128], we need [2 ; T ; nBands * 128]
        nChannels = x.shape[0]
        out = x.reshape((nChannels, x.shape[1], -1))
        out = self.layers(out)
        # Reshape back to [2; T; nBands; 128]
        out = out.reshape((nChannels, x.shape[1], len(generate_bandsplits()), band_features))
        return out + x

    def forward_recurrent(self, x: torch.Tensor):
        # X is [2; nBands; 128], we need [2 ; nBands * 128]
        nChannels = x.shape[0]
        out = x.reshape((nChannels, -1))
        out = self.layers(out)
        # Reshape back to [2; T; nBands; 128]
        out = out.reshape((nChannels, len(generate_bandsplits()), band_features))
        return out + x

class BandwiseNoop(nn.Module):
    def __init__(self):
        super(BandwiseNoop, self).__init__()

    def forward(self, x):
        return x

class BandwiseConv(nn.Module):
    def __init__(self):
        super(BandwiseConv, self).__init__()
        self.nBands = len(generate_bandsplits())
        self.layers = nn.Sequential(
            XXCInstanceNorm1d(self.nBands),
            nn.Conv1d(band_features, band_features, 3, stride = 1, padding = 'same'),
            nn.LeakyReLU(),
            nn.Conv1d(band_features, band_features, 3, stride = 1, padding = 'same'),
            nn.LeakyReLU(),
            nn.Conv1d(band_features, band_features, 3, stride = 1, padding = 'same'),
            nn.LeakyReLU(),
            nn.Conv1d(band_features, band_features, 3, stride = 1, padding = 'same'),
            nn.LeakyReLU(),
            nn.Conv1d(band_features, band_features, 3, stride = 1, padding = 'same'),
            nn.LeakyReLU(),
            nn.Conv1d(band_features, band_features, 3, stride = 1, padding = 'same')
        )

    def forward(self, x: torch.Tensor):
        # X is [2; T; nBands; 128], we want [2*T; 128; nBands]
        nChannels = x.shape[0]
        out = x.permute( (0, 1, 3, 2) )
        out = out.reshape( (-1, band_features, self.nBands) )
        out = self.layers(out)
        out = out.reshape( (nChannels, -1, band_features, self.nBands) )
        out = out.permute( (0, 1, 3, 2) )
        return out + x

    def forward_recurrent(self, x):
        # X is [2; nBands; 128], we want [2*T; 128; nBands]
        out = x.permute( (0, 2, 1) )
        out = self.layers(out)
        out = out.permute( (0, 1, 2) )
        return out + x


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
    #v = [
    #    (5, 0),
    #    (5, 5),
    #    (5, 10),
    #    (5, 15),
    #    (10, 20),
    #    (10, 30),
    #    (10, 40),
    #    (10, 50),
    #    (10, 60),
    #    (10, 70),
    #    (10, 80),
    #    (10, 90),
    #    (50, 100),
    #    (50, 150),
    #    (50, 200),
    #    (50, 250),
    #    (50, 300),
    #    (50, 350),
    #    (50, 400),
    #    (50, 450),
    #    (100, 500),
    #    (100, 600),
    #    (100, 700),
    #    (100, 800),
    #    (100, 900),
    #    (250, 1000),
    #    (250, 1250),
    #    (250, 1500),
    #]
    pos = 5
    # Split each "octave" in two
    mul = 2
    v = [(5, 0)]
    while pos < 2049:
        n = int(pos * mul)
        v += [(n - pos, pos)]
        pos += n - pos
    v.pop()

    pos = 0
    for x,y in v:
        assert y == pos
        pos += x
    v = [x[0] for x in v]
    v = v + [2049 - sum(v)]
    return v

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

        num_lstm_layers = 2
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
                nn.Linear(mask_estimation_mlp_hidden, mask_estimation_mlp_hidden),
                nn.LeakyReLU(),
                nn.Linear(mask_estimation_mlp_hidden, x * 2 * 2),
                nn.GLU(),
            ) for x in generate_bandsplits()])

    # Signal is 48kHz
    # pre-x is [2; T] where T is the number of samples
    # Caller do STFTs on the input, with a 4096 Window, and 512 hop length (so 88% overlap)
    # (So one sample will be seen 8 times)
    # Input `x` of this function is [2; F * 2; T/512] where F is the number of frequencies,
    # here 2049 * 2 to account for real + complex
    # From here on, we stop saying "T/512" but just T
    def forward(self, x):

        pshape("STFT", x.shape)
        bandsplit = generate_bandsplits()
        current_band_start = 0

        if merge_channels:
            m = x.mean(dim = 0)
            m = m.unsqueeze(0)
        else:
            m = x
        pshape("merged channels", m.shape)
        bands = m.split( [2 * i for i in generate_bandsplits()], dim = 1)

        # Now we have the bands, we can do the band specific processing
        band_outputs = []
        for i, band in enumerate(bands):
            # Permute the band to [2; T; F]
            band = band.permute((0, 2, 1))
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
            # band is [2; T; <filter size * 2>]
            pshape("Band back", i, band.shape)
            mask_estimations.append(band)
        mask_estimations = torch.cat(mask_estimations, 2)

        mask_estimations = mask_estimations.permute((0, 2, 1))

        pshape("Final mask is", mask_estimations.shape)
        pshape("x is ", x.shape)

        if merge_channels:
            # mask is [1 ; freqs], switch it to [freqs] to allow broadcasting
            mask_estimations = mask_estimations.squeeze(0)

        x = x * mask_estimations

        return x

    def forward_recurrent(self, x, state):
        pshape("STFT", x.shape)
        bandsplit = generate_bandsplits()
        current_band_start = 0

        if merge_channels:
            m = x.mean(dim = 0)
            m = m.unsqueeze(0)
        else:
            m = x
        pshape("merged channels", m.shape)
        bands = m.split( [2 * i for i in generate_bandsplits()], dim = 1)

        # Now we have the bands, we can do the band specific processing
        band_outputs = []
        for i, band in enumerate(bands):
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
                band_outputs, s = layer.forward_recurrent(band_outputs, state[2*state_i:2*state_i+2])
                state_i += 1
                new_state += [s]
            else:
                band_outputs = layer.forward_recurrent(band_outputs)
            pshape(band_outputs.shape)
        new_state = torch.cat(new_state, dim=0)

        bands_with_time_and_bands = band_outputs

        # Now we have the bands with time and bands, we can do the band specific processing
        mask_estimations = []
        for i, ogBandFc in enumerate(self.bandFCs):
            band = bands_with_time_and_bands[:, i, :]
            band = self.bandFCs_back[i](band)
            pshape("Band back pre", i, band.shape)
            # band is [2; T; <filter size * 2>]
            pshape("Band back", i, band.shape)
            mask_estimations.append(band)
        mask_estimations = torch.cat(mask_estimations, 1)

        pshape("Final mask is", mask_estimations.shape)
        pshape("x is ", x.shape)

        if merge_channels:
            # mask is [1 ; freqs], switch it to [freqs] to allow broadcasting
            mask_estimations = mask_estimations.squeeze(0)

        x = x * mask_estimations + rem

        return x, new_state
