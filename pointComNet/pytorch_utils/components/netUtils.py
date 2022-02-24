import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
import torch
import torch.nn.functional as F
import torch.nn.parallel
from torch.nn import Sequential as Seq, ReLU, BatchNorm1d as BN
import torch.nn as nn


class NetUtil:
    @staticmethod
    def SeqLinear(channels, batch_norm=True, activation="relu"):
        if batch_norm:
            if activation == "relu":
                return Seq(*[
                    Seq(torch.nn.Linear(channels[i - 1], channels[i]),
                        torch.nn.BatchNorm1d(channels[i]),
                        torch.nn.ReLU())
                    for i in range(1, len(channels))])
            else:
                return Seq(*[
                    Seq(torch.nn.Linear(channels[i - 1], channels[i]),
                        torch.nn.BatchNorm1d(channels[i]),
                        torch.nn.LeakyReLU(negative_slope=0.2))
                    for i in range(1, len(channels))])

        else:
            if activation == "relu":
                return Seq(*[
                    Seq(torch.nn.Linear(channels[i - 1], channels[i]),
                        torch.nn.ReLU())
                    for i in range(1, len(channels))])
            else:
                return Seq(*[
                    Seq(torch.nn.Linear(channels[i - 1], channels[i]),
                        torch.nn.LeakyReLU(negative_slope=0.2))
                    for i in range(1, len(channels))])

    @staticmethod
    def SeqPointNetConv1d(channels, batch_norm=True, active="relu"):
        if active == "relu":
            return Seq(*[
                Seq(torch.nn.Conv1d(channels[i - 1], channels[i], 1, bias=False),
                    torch.nn.BatchNorm1d(channels[i]),
                    torch.nn.ReLU())
                for i in range(1, len(channels))])
        else:
            return Seq(*[
                Seq(torch.nn.Conv1d(channels[i - 1], channels[i], 1, bias=False),
                    torch.nn.BatchNorm1d(channels[i]),
                    torch.nn.LeakyReLU(negative_slope=0.2))
                for i in range(1, len(channels))])

    @staticmethod
    def SeqPointNetConv2d(channels, batch_norm=True):
        return Seq(*[
            Seq(torch.nn.Conv2d(channels[i - 1], channels[i], 1, bias=False),
                torch.nn.BatchNorm2d(channels[i]),
                torch.nn.ReLU())
            for i in range(1, len(channels))])

    @staticmethod
    def SeqPointNetConv3d(channels, batch_norm=True):
        return Seq(*[
            Seq(torch.nn.Conv3d(channels[i - 1], channels[i], 1, bias=False),
                torch.nn.BatchNorm3d(channels[i]),
                torch.nn.ReLU())
            for i in range(1, len(channels))])

    @staticmethod
    def SetPointDGCNN(channels, batch_norm=True):
        return Seq(*[
            Seq(torch.nn.Conv2d(channels[i - 1] * 2, channels[i], 1, bias=False),
                torch.nn.BatchNorm2d(channels[i]),
                torch.nn.LeakyReLU(negative_slope=0.2))
            for i in range(1, len(channels))])

    @staticmethod
    def classification_loss(pred, gold, smoothing=True):
        ''' Calculate cross entropy loss, apply label smoothing if needed. '''

        gold = gold.contiguous().view(-1)
        if smoothing:
            eps = 0.2
            n_class = pred.size(1)

            one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)  # creat a one-hot vector
            one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
            log_prb = F.log_softmax(pred, dim=1)

            loss = -(one_hot * log_prb).sum(dim=1).mean()
        else:
            loss = F.cross_entropy(pred, gold, reduction='mean')

        return loss


class CPointNet(nn.Module):
    def __init__(self, channels):
        super(CPointNet, self).__init__()
        self.conv1d = nn.ModuleList([Seq(torch.nn.Conv1d(channels[i - 1], channels[i], 1, bias=False),
                                         torch.nn.BatchNorm1d(channels[i]),
                                         torch.nn.ReLU()) for i in range(1, len(channels))])

    def forward(self, x):
        out = []
        for layer in self.conv1d:
            x = layer(x)
            out.append(x)
        cmlp = torch.cat(out, dim=1)
        return cmlp


class PointNet(nn.Module):
    def __init__(self, channels, activation="relu"):
        super(PointNet, self).__init__()
        # self.conv = Seq(*[Seq(torch.nn.Conv1d(channels[i - 1], channels[i], 1, bias=False),
        #                       torch.nn.BatchNorm1d(channels[i]),
        #                       torch.nn.ReLU())
        #                   for i in range(1, len(channels[:-1]))])
        #
        # self.conv_last = nn.Sequential(torch.nn.Conv1d(channels[-2], channels[-1], 1, bias=False),
        #                                torch.nn.BatchNorm1d(channels[-1]))
        self.conv = NetUtil.SeqPointNetConv1d(channels=channels, active=activation)

    def forward(self, x):
        x = self.conv(x)
        x = torch.max(x, -1)[0]
        return x


class CPointNetLinear(nn.Module):
    def __init__(self, channels):
        super(CPointNetLinear, self).__init__()
        self.conv1d = nn.ModuleList([Seq(torch.nn.Conv1d(channels[i - 1], channels[i], 1, bias=False),
                                         torch.nn.BatchNorm1d(channels[i]),
                                         torch.nn.ReLU()) for i in range(1, len(channels))])

    def forward(self, x):
        out = []
        B, N, D = x.shape
        for layer in self.conv1d:
            x = layer(x)
            z_hat = torch.max(x, 2, keepdim=True)[0]
            out.append(z_hat)
        cmlp = torch.cat(out, dim=1)
        cmlp = cmlp.view(B, -1)
        return cmlp
