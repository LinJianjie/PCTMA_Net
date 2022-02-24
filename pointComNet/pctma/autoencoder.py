import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from pointComNet.utils.loss_function import EMDLoss, CDLoss
import sys
import os
from torch.autograd import Variable
from pathlib import Path

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))
from pointComNet.pytorch_utils.components.netUtils import NetUtil
from pointComNet.pytorch_utils.components.dgcnn.dgcnn_utils import DGCNN
import copy
import numpy as np


class PointGenCon(nn.Module):
    def __init__(self, bottleneck_size=1024, output=3, use_tanh=True):
        super(PointGenCon, self).__init__()
        self.bottleneck_size = bottleneck_size
        channels = [self.bottleneck_size, self.bottleneck_size, self.bottleneck_size // 2, self.bottleneck_size // 4]
        self.conv_module = nn.ModuleList([nn.Sequential(nn.Conv1d(channels[i - 1], channels[i], kernel_size=1),
                                                        nn.BatchNorm1d(channels[i]),
                                                        nn.ReLU()) for i in range(1, len(channels))])

        self.conv4 = torch.nn.Conv1d(self.bottleneck_size // 4, output, 1)
        self.use_tanh = use_tanh
        self.th = nn.Tanh()

    def forward(self, x):
        for layer in self.conv_module:
            x = layer(x)
        x = self.conv4(x)
        if self.use_tanh:
            x = self.th(x)
        return x


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class NetMADecoder(nn.Module):
    def __init__(self, args, num_points, channels, bottleneck_size=1024):
        super(NetMADecoder, self).__init__()
        self.num_points = num_points
        self.args = args
        # self.De_layers = NetUtil.SeqLinear(channels)
        # self.fc3 = nn.Linear(channels[-1], num_points * 3)
        self.additional_size = 2
        self.bottleneck_size = bottleneck_size
        self.n_primitives = self.args["n_primitives"]
        self.fine_factor = self.args["fine_factor"]
        self.num_points_3 = 128
        self.num_points_2 = 512
        self.num_points_1 = self.num_points * self.fine_factor

        print("the output num_points: ", self.num_points_1)
        # self.num_points_1 = 2048

        self.bottleneck_size_1 = channels
        self.bottleneck_size_2 = 512
        self.bottleneck_size_3 = 256
        self.use_atlas = self.args["use_atlas"]
        print("--> self.use_atlas: ", self.use_atlas)
        self.use_pointModule = self.args["use_pointModule"]
        pointModule = ["linear_Feature", "atlas", "foldnet", "multi_head_atlas"]
        print("   --> use point_moudle[1=atlas|3=multi_head_atlas]: ",
              pointModule[self.use_pointModule])
        self.mh_case = False
        if self.use_pointModule == 3:
            self.mh_case = False
            print("--> use full mh_case: ", self.mh_case)

        self.use_rand_grid = False
        if not self.use_rand_grid:
            self.grid_size = int(np.sqrt(self.num_points_1 // self.n_primitives))
            if not self.mh_case:
                self.num_points_1 = self.grid_size ** 2 * self.n_primitives
            print("the output num_points modified: ", self.num_points_1 + 2048)
            if self.use_pointModule == 2:
                self.grid_size = 4
            self.grid_scale = self.args["grid_scale"]
            print("---> grid_size: ", self.grid_size, " grid_scale: ", self.grid_scale)
        print("---> use rand_grid: ", self.use_rand_grid, " use mesh grid: ", not self.use_rand_grid)

        if self.use_atlas:
            self.atlas_primitives = []
            if self.use_pointModule == 1:
                self.decoder_1 = nn.ModuleList(
                    [PointGenCon(bottleneck_size=self.additional_size + self.bottleneck_size_1) for i in
                     range(0, self.n_primitives)])
            if self.use_pointModule == 3:
                self.linear_project = nn.Sequential(nn.Conv1d(self.additional_size + self.bottleneck_size_1,
                                                              self.additional_size + self.bottleneck_size_1,
                                                              kernel_size=1),
                                                    nn.BatchNorm1d(self.additional_size + self.bottleneck_size_1),
                                                    nn.ReLU())
                self.decoder_1 = nn.ModuleList(
                    [PointGenCon(bottleneck_size=self.additional_size + self.bottleneck_size_1) for i in
                     range(0, self.n_primitives)])

    def forward(self, x):
        x_1 = x
        if self.use_pointModule == 1:
            pc1_xyz = self.atlasNet(x_1, self.decoder_1, out_num=self.num_points_1)
            return None, None, pc1_xyz
        if self.use_pointModule == 3:
            pc1_xyz = self.multi_head_atlasNet(x_1, self.decoder_1, out_num=self.num_points_1,
                                               linear_project=self.linear_project)
            return None, None, pc1_xyz

    def atlasNet(self, x, decoder, out_num):
        outs = []
        add_rand = False
        for i in range(0, self.n_primitives):
            if self.use_rand_grid:
                rand_grid = Variable(
                    torch.cuda.FloatTensor(x.size(0), self.additional_size, out_num // self.n_primitives))
                rand_grid.data.uniform_(0, 1)
            else:
                grid_size = self.grid_size
                grid_row = torch.linspace(-1 * self.grid_scale, self.grid_scale, self.grid_size).to(x.device)
                grid_col = torch.linspace(-1 * self.grid_scale, self.grid_scale, self.grid_size).to(x.device)
                grid = torch.meshgrid(grid_row, grid_col)
                grid = torch.reshape(torch.stack(grid, dim=2), (-1, 2)).unsqueeze(0)
                grid = grid.repeat(x.size(0), 1, 1)
                rand_grid = grid.transpose(2, 1)
            y = x.unsqueeze(2).expand(x.size(0), x.size(1), rand_grid.size(2)).contiguous()
            y = torch.cat((rand_grid, y), 1).contiguous()
            outs.append(decoder[i](y))
        self.atlas_primitives = outs
        outs = torch.cat(outs, 2).contiguous()
        return outs.transpose(2, 1).contiguous()

    def multi_head_atlasNet(self, x, decoder, out_num, linear_project):
        outs = []
        if self.mh_case:
            grid_row = torch.linspace(-1 * self.grid_scale, self.grid_scale, int(np.ceil(np.sqrt(out_num)))).cuda()
            grid_col = torch.linspace(-1 * self.grid_scale, self.grid_scale, int(np.ceil(np.sqrt(out_num)))).cuda()
            grid = torch.meshgrid(grid_row, grid_col)
            grid = torch.reshape(torch.stack(grid, dim=2), (-1, 2)).unsqueeze(0)
            grid = grid.repeat(x.size(0), 1, 1).transpose(2, 1)
            grid = grid[:, :, :out_num]
        else:
            grid_row = torch.linspace(-1 * self.grid_scale, self.grid_scale, self.grid_size).cuda()
            grid_col = torch.linspace(-1 * self.grid_scale, self.grid_scale, self.grid_size).cuda()
            grid = torch.meshgrid(grid_row, grid_col)
            grid = torch.reshape(torch.stack(grid, dim=2), (-1, 2)).unsqueeze(0)
            grid = grid.repeat(x.size(0), self.n_primitives, 1).transpose(2, 1)

        y = x.unsqueeze(2).expand(x.size(0), x.size(1), grid.size(2)).contiguous()
        y = torch.cat((grid, y), 1).contiguous()  # 32*1026*out_num
        y = linear_project(y)  # 32*1026*out_num
        new_features = y.chunk(self.n_primitives, dim=-1)
        for i in range(0, self.n_primitives):
            outs.append(decoder[i](new_features[i]))
        self.atlas_primitives = outs
        outs = torch.cat(outs, 2).transpose(2, 1).contiguous()
        return outs
