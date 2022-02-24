import os
import sys
import torch
import unittest

from torch.autograd import gradcheck

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir)))
from externs_tools.fps import fps

if __name__ == '__main__':
    cuda_index = "cuda:0"
    device = torch.device(cuda_index if (torch.cuda.is_available()) else "cpu")
    # x = torch.tensor([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]]).cuda()
    x = torch.rand(100, 3, device=torch.device('cuda:0'))
    print(x.size(0))
    batch = torch.zeros(100, device=torch.device('cuda:0'), dtype=torch.int64)
    print(batch.numel())
    index = fps(x, batch, ratio=0.5, random_start=False)
    print(index)
    index = fps(x, batch, ratio=0.5, random_start=False)
    print(index)
    index = fps(x, batch, ratio=0.5, random_start=False)
    print(index)
