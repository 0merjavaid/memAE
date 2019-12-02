from __future__ import absolute_import, print_function
import torch
from torch import nn


class Encoder(nn.Module):

    def __init__(self, chnum_in=3, mem_dim=64, shrink_thres=0.0025):
        super(Encoder, self).__init__()
        print('Encoder')
        self.chnum_in = chnum_in
        feature_num = 16
        feature_num_2 = 32
        feature_num_x2 = 240
        self.encoder = nn.Sequential(
            nn.Conv2d(self.chnum_in, feature_num_2, (3, 3),
                      stride=2, padding=1),
            nn.BatchNorm2d(feature_num_2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_num_2, feature_num, (3, 3),
                      stride=2, padding=1),
            nn.BatchNorm2d(feature_num),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_num, feature_num_x2, (3, 3),
                      ),
            nn.BatchNorm2d(feature_num_x2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_num_x2, feature_num_x2, (3, 3),
                      stride=2),
            nn.BatchNorm2d(feature_num_x2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        f = self.encoder(x)
        return f


class Decoder(nn.Module):

    def __init__(self, chnum_in=3):
        super(Decoder, self).__init__()
        print('Decoder')
        self.chnum_in = chnum_in
        feature_num = 16
        feature_num_2 = 32
        feature_num_x2 = 240
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(feature_num_x2, feature_num_x2, (3, 3), stride=(2, 2), padding=(1, 1),
                               output_padding=(1, 1)),
            nn.BatchNorm2d(feature_num_x2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(feature_num_x2, feature_num, (3, 3), stride=(2, 2), padding=(1, 1),
                               output_padding=(1, 1)),
            nn.BatchNorm2d(feature_num),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(feature_num, feature_num_2, (3, 3), stride=(2, 2), padding=(1, 1),
                               ),
            nn.BatchNorm2d(feature_num_2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(feature_num_2, feature_num_2, (3, 3), stride=(2, 2), padding=(1, 1),
                               output_padding=(1, 1)),

            nn.BatchNorm2d(feature_num_2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(feature_num_2, self.chnum_in, (3, 3), stride=(2, 2), padding=(1, 1),
                               output_padding=(1, 1))

        )

    def forward(self, x):
        f = self.decoder(x)
        return f
