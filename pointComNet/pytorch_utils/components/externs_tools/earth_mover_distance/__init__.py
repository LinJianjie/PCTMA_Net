import torch
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir)))
chamferFunction_build = "lib/python3.6/site-packages/emd-1.0.0-py3.6-linux-x86_64.egg/"
chamferFunction_path = os.path.join(os.path.join(os.path.dirname(__file__)),
                                    "../../../extensions_build/earch_mover_distance",
                                    chamferFunction_build)
sys.path.append(chamferFunction_path)
import emd


class EarthMoverDistanceFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xyz1, xyz2):
        xyz1 = xyz1.contiguous()
        xyz2 = xyz2.contiguous()
        assert xyz1.is_cuda and xyz2.is_cuda, "Only support cuda currently."
        match = emd.approxmatch_forward(xyz1, xyz2)
        cost = emd.matchcost_forward(xyz1, xyz2, match)
        ctx.save_for_backward(xyz1, xyz2, match)
        return cost

    @staticmethod
    def backward(ctx, grad_cost):
        xyz1, xyz2, match = ctx.saved_tensors
        grad_cost = grad_cost.contiguous()
        grad_xyz1, grad_xyz2 = emd.matchcost_backward(grad_cost, xyz1, xyz2, match)
        return grad_xyz1, grad_xyz2


def earth_mover_distance(xyz1, xyz2, transpose=True):
    """Earth Mover Distance (Approx)
    Args:
        xyz1 (torch.Tensor): (b, 3, n1)
        xyz2 (torch.Tensor): (b, 3, n1)
        transpose (bool): whether to transpose inputs as it might be BCN format.
            Extensions only support BNC format.
    Returns:
        cost (torch.Tensor): (b)
    """
    if xyz1.dim() == 2:
        xyz1 = xyz1.unsqueeze(0)
    if xyz2.dim() == 2:
        xyz2 = xyz2.unsqueeze(0)
    if transpose:
        xyz1 = xyz1.transpose(1, 2)
        xyz2 = xyz2.transpose(1, 2)
    cost = EarthMoverDistanceFunction.apply(xyz1, xyz2)
    return cost


class EarthMoverDistance(torch.nn.Module):
    def __init__(self, ignore_zeros=False):
        super(EarthMoverDistance, self).__init__()
        self.ignore_zeros = ignore_zeros

    def forward(self, xyz1, xyz2, transpose=False):
        """Earth Mover Distance (Approx)
        Args:
            xyz1 (torch.Tensor): (b, n1, 3)
            xyz2 (torch.Tensor): (b, n1, 3)
            transpose (bool): whether to transpose inputs as it might be BCN format.
                Extensions only support BNC format.
        Returns:
            cost (torch.Tensor): (b)
        """
        if xyz1.dim() == 2:
            xyz1 = xyz1.unsqueeze(0)
        if xyz2.dim() == 2:
            xyz2 = xyz2.unsqueeze(0)
        if transpose:
            xyz1 = xyz1.transpose(1, 2)
            xyz2 = xyz2.transpose(1, 2)
        cost = EarthMoverDistanceFunction.apply(xyz1, xyz2)
        return torch.mean(cost)
