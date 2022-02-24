import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from components.netUtils import NetUtil


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)  # (batch_size, num_points, k)
    idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()
    # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature


def edge_conv(x, k, layer):
    x = get_graph_feature(x, k=k)
    x = layer(x)
    x1 = x.max(dim=-1, keepdim=False)[0]
    return x1


class DGCNN(nn.Module):
    def __init__(self, k=20, channels=None):
        super(DGCNN, self).__init__()
        self.k = k
        print(" knn: ", self.k)
        # self.emb_dims = 1024
        # self.bn1 = nn.BatchNorm2d(64)
        # self.bn2 = nn.BatchNorm2d(64)
        # self.bn3 = nn.BatchNorm2d(128)
        # self.bn4 = nn.BatchNorm2d(256)
        # self.bn5 = nn.BatchNorm1d(self.emb_dims)
        #
        # self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
        #                            self.bn1,
        #                            nn.LeakyReLU(negative_slope=0.2))
        # self.conv2 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
        #                            self.bn2,
        #                            nn.LeakyReLU(negative_slope=0.2))
        # self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
        #                            self.bn3,
        #                            nn.LeakyReLU(negative_slope=0.2))
        # self.conv4 = nn.Sequential(nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),
        #                            self.bn4,
        #                            nn.LeakyReLU(negative_slope=0.2))
        self.edge_layers = nn.ModuleList([
            nn.Sequential(torch.nn.Conv2d(channels[i - 1] * 2, channels[i], kernel_size=1, bias=False),
                          torch.nn.BatchNorm2d(channels[i]),
                          torch.nn.LeakyReLU(negative_slope=0.2))
            for i in range(1, len(channels[:-1]))])
        # self.edge_layers = NetUtil.SetPointDGCNN(channels[:-1])
        self.use_two_layer = False
        if self.use_two_layer:
            self.first_layer = channels[-1] // 2
            self.conv_fuse = nn.Sequential(nn.Conv1d(sum(channels[1:-1]), self.first_layer, kernel_size=1, bias=False),
                                           nn.BatchNorm1d(self.first_layer),
                                           nn.LeakyReLU(negative_slope=0.2))
            self.conv5 = nn.Sequential(nn.Conv1d(self.first_layer, channels[-1], kernel_size=1, bias=False),
                                       nn.BatchNorm1d(channels[-1]),
                                       nn.LeakyReLU(negative_slope=0.2))
        else:
            self.conv5 = nn.Sequential(nn.Conv1d(sum(channels[1:-1]), channels[-1], kernel_size=1, bias=False),
                                       nn.BatchNorm1d(channels[-1]),
                                       nn.LeakyReLU(negative_slope=0.2))

        self.adaptive_max_pool1d = nn.AdaptiveMaxPool1d(output_size=1)
        #self.adaptive_avg_pool1d = nn.AdaptiveAvgPool1d(output_size=1)

    def forward(self, x):
        batch_size = x.size(0)
        hidden_state = []
        for layer in self.edge_layers:
            x = edge_conv(x, self.k, layer)
            hidden_state.append(x)
        # x = get_graph_feature(x, k=self.k)
        # x = self.conv1(x)
        # x1 = x.max(dim=-1, keepdim=False)[0]
        #
        # x = get_graph_feature(x1, k=self.k)
        # x = self.conv2(x)
        # x2 = x.max(dim=-1, keepdim=False)[0]
        #
        # x = get_graph_feature(x2, k=self.k)
        # x = self.conv3(x)
        # x3 = x.max(dim=-1, keepdim=False)[0]
        #
        # x = get_graph_feature(x3, k=self.k)
        # x = self.conv4(x)
        # x4 = x.max(dim=-1, keepdim=False)[0]
        x = torch.cat(hidden_state, dim=1)
        # x = torch.cat((x1, x2, x3, x4), dim=1)
        if self.use_two_layer:
            x = self.conv_fuse(x)
            x = self.conv5(x)
        else:
            x = self.conv5(x)
        # x2 = self.adaptive_max_pool1d(x).view(batch_size, -1)
        # x2 = x.max(dim=-1, keepdim=False)[0]
        x2 = self.adaptive_max_pool1d(x).view(batch_size, -1)
        # x3 = torch.cat([x1, x2], dim=-1)
        return x2
