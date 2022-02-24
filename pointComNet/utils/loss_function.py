"""
Collection of different loss functions
"""

import torch.nn as nn
import torch
from geomloss import SamplesLoss
# Fix absolute imports
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

# Additionally remove the current file's directory from sys.path

# Absolute imports
from pointComNet.pytorch_utils.components.externs_tools.chamfer_distance_topnet import ChamferDistance
from pointComNet.pytorch_utils.components.externs_tools.earth_mover_distance import EarthMoverDistance
from pointComNet.pytorch_utils.components.externs_tools.msn_emd import emdModule
from pointComNet.pytorch_utils.components.externs_tools.gridding import Gridding, GriddingReverse
from pointComNet.pytorch_utils.components.externs_tools.cubic_feature_sampling import CubicFeatureSampling


# from pointComNet.pytorch_utils.components.externs_tools.chamfer_distance_jit import ChamferDistance


class EMDLossAlt(nn.Module):
    """
    Earth Movers Distance or Wasserstein Loss - only available for CUDA
    Returns the mean EMD for the batch based on the Sinkhorn Distance approximation
    """

    def __init__(self):
        super(EMDLossAlt, self).__init__()
        self.sinkhorn = SamplesLoss(loss="sinkhorn", p=2, blur=.05)  # Expects n x 3 matrices

    def forward(self, pcs1, pcs2):
        """
        :param pcs1: expects [batch_size x num_points x 3] PyTorch tensor
        :param pcs2: expects [batch_size x num_points x 3] PyTorch tensor, must be same shape as pcs1
        :return: loss as float
        """
        bs = pcs1.shape[0]  # batch size
        pcs1 = pcs1.permute(0, 2, 1)
        pcs2 = pcs2.permute(0, 2, 1)
        loss = torch.empty((bs,), dtype=torch.float)
        for i in range(bs):
            loss[i] = self.sinkhorn(pcs1[i], pcs2[i])
        # tt=torch.mean(loss)
        # print("tt: ",tt.shape)
        return torch.mean(loss)


class EMDLoss(nn.Module):
    def __init__(self, use_msn=False):
        super(EMDLoss, self).__init__()
        print("using use_msn emd_loss: ", use_msn)
        if use_msn:
            self.emd_dist = emdModule()
        else:
            self.emd_dist = EarthMoverDistance()

    def forward(self, pcs1, pcs2):
        # assert pcs1.shape == pcs2.shape
        assert pcs1.shape[2] == 3
        return self.emd_dist(pcs1, pcs2)


class CDLoss(nn.Module):
    """
    Chamfer Distance Batch Loss - only available for CUDA, no cpu support given
    Computes CD based average loss for batch of pcs
    Adapted from https://github.com/chrdiller/pyTorchChamferDistance
    """

    def __init__(self):
        super(CDLoss, self).__init__()
        self.chamfer_dist = ChamferDistance()

    def forward(self, pcs1, pcs2):
        """
        :param pcs1: expects [batch_size x num_points x 3] PyTorch tensor
        :param pcs2: expects [batch_size x num_points x 3] PyTorch tensor, must be same shape as pcs1
        :return: loss as float tensor
        """
        # Squared distance between each point in pcs1 to its nearest neighbour in pcs2 and vice versa
        # print("pcs1: ",pcs1.shape)
        # print("pcs2: ", pcs2.shape)
        # pcs1 = pcs1.permute(0, 2, 1)
        # pcs2 = pcs2.permute(0, 2, 1)
        # assert pcs1.shape == pcs2.shape
        assert pcs1.shape[2] == 3
        mean_dist, dist1, dist2 = self.chamfer_dist(pcs1, pcs2)
        return mean_dist


class CDLossEval(nn.Module):
    def __init__(self):
        super(CDLossEval, self).__init__()
        self.chamfer_dist = ChamferDistance()

    def forward(self, pcs1, pcs2):
        assert pcs1.shape[2] == 3
        mean_dist, dist1, dist2 = self.chamfer_dist(pcs1, pcs2)
        return mean_dist, dist1, dist2


class CubicFeatureSampling_function(nn.Module):
    def __init__(self):
        super(CubicFeatureSampling_function, self).__init__()
        self.cubic_feature_sampling = CubicFeatureSampling()

    def forward(self, ptcloud, cubic_features, neighborhood_size=1):
        return self.cubic_feature_sampling(ptcloud, cubic_features, neighborhood_size)


class Gridding_function(nn.Module):
    def __init__(self, scale=1):
        super(Gridding_function, self).__init__()
        self.gridding = Gridding(scale=scale)

    def forward(self, ptcloud):
        return self.gridding(ptcloud)


class Gridding_Reverse_function(nn.Module):
    def __init__(self, scale=1):
        super(Gridding_Reverse_function, self).__init__()
        self.gridding_reverse = GriddingReverse(scale=scale)

    def forward(self, grid):
        return self.gridding_reverse(grid)


def l2_loss(input, target, size_average=True):
    """ L2 Loss without reduce flag.
    Args:
        input (FloatTensor): Input tensor
        target (FloatTensor): Output tensor
    Returns:
        [FloatTensor]: L2 distance between input and output
    """
    if size_average:
        return torch.mean(torch.pow((input - target), 2))
    else:
        return torch.pow((input - target), 2)


if __name__ == '__main__':
    """
    Structural testing purpose only
    """
    x = torch.randn(32, 1024, 3, requires_grad=True).cuda()
    y = torch.randn(32, 1024, 3).cuda()
    print(x.size(), y.size())

    print("Chamfer Loss:")
    loss_metric = CDLoss()
    print(loss_metric(x, y))

    print("EMD Loss:")
    loss_metric = EMDLoss()
    print(loss_metric(x, y))
