import os
import sys
import torch
import unittest

from torch.autograd import gradcheck

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir)))
from externs_tools.knn import knn, knn_graph

if __name__ == '__main__':
    x = torch.tensor([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]])
    batch = torch.tensor([0, 0, 0, 0])
    edge_index = knn_graph(x, k=2, batch=batch, loop=False)
    print(edge_index)
