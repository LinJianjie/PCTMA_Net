import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from components.pointNetmodel.pc_common import *
from components.torch_cluster_sampling import *
from components.netUtils import NetUtil
from components.dataUtils import *


class PointNetSetAbstraction(nn.Module):
    def __init__(self, ratio, radius, nsample, mlp_list, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.ratio = ratio
        self.radius = radius
        self.nsample = nsample
        self.pointNet = NetUtil.SeqPointNetConv2d(channels=mlp_list)
        self.group_all = group_all

    def forward(self, xyz, features):
        """
        Input:
            xyz: input points position data, [B, C, N] --> waypoints
            points: input points data, [B, D, N] --> feature from nn
        Return:
            new_xyz: sampled points position data, [B, C, S] --> fps centrid
            new_points_concat: sample points feature data, [B, D', S] --> new features vector
        """
        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, features)
        else:
            new_xyz, _, _ = FPS(x=xyz, ratio=self.ratio).get()
            group_idx = query_ball_point(self.radius, self.nsample, xyz, new_xyz)
            new_points = DataUtils.index_points(features, group_idx)
        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]
        new_points = new_points.permute(0, 3, 2, 1)  # [B, C+D, nsample,npoint]
        new_points = self.pointNet(new_points)
        new_points = torch.max(new_points, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points


class PointNetSetAbstractionMsg(nn.Module):
    def __init__(self, ratio, radius_list, max_sample_list, mlp_list):
        super(PointNetSetAbstractionMsg, self).__init__()
        self.ratio = ratio
        self.radius_list = radius_list
        self.max_sample_list = max_sample_list
        self.msgPointNet = nn.ModuleList()
        # self.conv_blocks = nn.ModuleList()
        # self.bn_blocks = nn.ModuleList()
        # for i in range(len(mlp_list)):
        #     convs = nn.ModuleList()
        #     bns = nn.ModuleList()
        #     last_channel = in_channel + 3
        #     for out_channel in mlp_list[i]:
        #         convs.append(nn.Conv2d(last_channel, out_channel, 1))
        #         bns.append(nn.BatchNorm2d(out_channel))
        #         last_channel = out_channel
        #     self.conv_blocks.append(convs)
        #     self.bn_blocks.append(bns)
        for channels in mlp_list:
            pN = NetUtil.SeqPointNetConv2d(channels=channels)
            self.msgPointNet.append(pN)

    def forward(self, xyz, features):
        """
        Input:
            xyz: input points position data, [B, N, C] --> C is feature size, N ist the number of way
            points: input points data, [B, N, D]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        B, N, C = xyz.shape
        new_xyz, _, _ = FPS(x=xyz, ratio=self.ratio).get()
        new_points_list = []
        for i, radius in enumerate(self.radius_list):
            K = self.max_sample_list[i]
            group_idx = query_ball_point(radius, K, xyz, new_xyz)
            grouped_points = DataUtils.index_points(features, group_idx)
            grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, D, K, S]
            # for j in range(len(self.conv_blocks[i])):
            #     conv = self.conv_blocks[i][j]
            #     bn = self.bn_blocks[i][j]
            #     grouped_points = F.relu(bn(conv(grouped_points)))
            grouped_points = self.msgPointNet[i](grouped_points)
            new_points = torch.max(grouped_points, 2)[0]  # [B, D', S]
            new_points_list.append(new_points)
        new_xyz = new_xyz.permute(0, 2, 1)
        new_points_concat = torch.cat(new_points_list, dim=1)
        return new_xyz, new_points_concat


class PointNetFeaturePropagation(nn.Module):
    # TODO
    def __init__(self, mlp_list):
        super(PointNetFeaturePropagation, self).__init__()
        self.pointNet = NetUtil.SeqLinear(mlp_list)

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)

        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = DataUtils.square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(DataUtils.index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        new_points = self.pointNet(new_points)
        return new_points


if __name__ == '__main__':
    x = torch.tensor([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]])
    batch = torch.tensor([0, 0, 0, 0])
    index = fps(x, batch, ratio=0.5, random_start=False)
    print(index)
