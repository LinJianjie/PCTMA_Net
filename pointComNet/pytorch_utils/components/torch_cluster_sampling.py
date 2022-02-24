import torch
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from components.externs_tools.fps import fps
from components.externs_tools.knn import knn
from components.externs_tools.radiusnn import radius
import copy


class FPS:
    def __init__(self, x, ratio):
        self.original_x = copy.deepcopy(x)
        self.x = copy.deepcopy(x)
        self.ratio = ratio
        self.batch_size = 1
        if x.dim() == 3:
            self.batch_size, self.N, self.dim = x.shape
            self.x = self.x.reshape(-1, self.dim)
        else:
            self.N = self.x.shape[0]
            self.dim = self.x.shape[1]

        batch_index = torch.arange(self.batch_size, device=self.x.device)
        self.batch_fps = batch_index.repeat(self.N, 1).transpose(0, 1).reshape(-1)

    def get(self):
        fps_idx = fps(self.x, self.batch_fps, self.ratio)
        fps_size = int(fps_idx.shape[0] / self.batch_size)
        fps_center = self.x[fps_idx].reshape(self.batch_size, -1, self.dim)
        return fps_center, self.x, fps_idx


def farthest_point_sampling(x, ratio):
    if ratio > 1:
        ratio = ratio / x.shape[1]
    original_x = x.clone()
    ratio = ratio
    batch_size = 1
    if x.dim() == 3:
        batch_size, N, dim = x.shape
        x = x.reshape(-1, dim)
    else:
        N = x.shape[0]
        dim = x.shape[1]

    batch_index = torch.arange(batch_size, device=x.device)
    batch_fps = batch_index.repeat(N, 1).transpose(0, 1).reshape(-1)
    fps_idx = fps(x, batch_fps, ratio)
    fps_size = int(fps_idx.shape[0] / batch_size)
    fps_center = x[fps_idx].reshape(batch_size, -1, dim)
    return fps_center, x, fps_idx


def K_NN(points, maxKnn, queryPoints):
    points = points.transpose(2, 1)
    queryPoints = queryPoints.transpose(2, 1)
    pointCRange = points.reshape(-1, points.shape[2])
    batchIndex = torch.arange(points.shape[0])
    batch = batchIndex.repeat(points.shape[1], 1).transpose(0, 1).reshape(-1).to(points.device)
    batch_query = batchIndex.repeat(queryPoints.shape[1], 1).transpose(0, 1).reshape(-1).to(points.device)
    queryPointsCRange = queryPoints.reshape(-1, queryPoints.shape[2])
    assign_index = knn(pointCRange, queryPointsCRange, maxKnn, batch, batch_query)
    indices = assign_index[1, :].reshape(queryPoints.shape[0], queryPoints.shape[1], maxKnn)
    indices = indices % points.shape[1]
    return indices


class RandomPointSampling(torch.nn.Module):
    def __init__(self, n_points):
        super(RandomPointSampling, self).__init__()
        self.n_points = n_points

    def forward(self, pred_cloud, partial_cloud=None):
        if partial_cloud is not None:
            pred_cloud = torch.cat([partial_cloud, pred_cloud], dim=1)

        _ptcloud = torch.split(pred_cloud, 1, dim=0)
        ptclouds = []
        for p in _ptcloud:
            non_zeros = torch.sum(p, dim=2).ne(0)
            p = p[non_zeros].unsqueeze(dim=0)
            n_pts = p.size(1)
            if n_pts < self.n_points:
                rnd_idx = torch.cat([torch.randint(0, n_pts, (self.n_points,))])
            else:
                rnd_idx = torch.randperm(p.size(1))[:self.n_points]
            ptclouds.append(p[:, rnd_idx, :])

        return torch.cat(ptclouds, dim=0).contiguous()


class BkNN:  # B,N,D
    def __init__(self, points, maxKnn, includeSelf=False, queryPoints=None):
        if ~includeSelf:
            self.maxKnn = maxKnn + 1
        else:
            self.maxKnn = maxKnn
        self.points = points
        self.pointCRange = self.points.reshape(-1, self.points.shape[2])
        batchIndex = torch.arange(self.points.shape[0])
        batch = batchIndex.repeat(self.points.shape[1], 1).transpose(0, 1).reshape(-1).to(points.device)
        # batch = batchIndex.repeat(self.points.shape[1], 1).transpose(0, 1).reshape(-1).cuda()
        if queryPoints is not None:
            self.queryPoints = queryPoints
            batch_query = batchIndex.repeat(queryPoints.shape[1], 1).transpose(0, 1).reshape(-1).to(points.device)
            # batch_query = batchIndex.repeat(queryPoints.shape[1], 1).transpose(0, 1).reshape(-1).cuda()
            queryPointsCRange = self.queryPoints.reshape(-1, queryPoints.shape[2])
            assign_index = knn(self.pointCRange, queryPointsCRange, self.maxKnn, batch, batch_query)
        else:
            assign_index = knn(self.pointCRange, self.pointCRange, self.maxKnn, batch, batch)
        localRegion = self.pointCRange[assign_index[1, :], :]
        # B*N'*k*dim
        if queryPoints is not None:
            self.cluster = localRegion.reshape(self.queryPoints.shape[0], self.queryPoints.shape[1], self.maxKnn,
                                               self.queryPoints.shape[2])
            self.indices = assign_index[1, :].reshape(self.queryPoints.shape[0], self.queryPoints.shape[1], self.maxKnn)
        else:
            self.cluster = localRegion.reshape(self.points.shape[0], self.points.shape[1], self.maxKnn,
                                               self.points.shape[2])
            self.indices = assign_index[1, :].reshape(self.points.shape[0], self.points.shape[1], self.maxKnn)
        self.indices = self.indices % self.points.shape[1]

        if ~includeSelf:
            self.remove_self_loop()

    def remove_self_loop(self):
        self.cluster = self.cluster[:, :, 1:, :]
        self.indices = self.indices[:, :, 1:]

    def query(self, index):
        return self.cluster[:, index, :, :]

    def queryIndics(self, index):
        return self.query(index), self.indices[:, index, :]


class BraidusNN:
    def __init__(self):
        pass

    def get(self):
        pass


if __name__ == '__main__':
    x = torch.rand(3, 15, 3)
    my_fps = FPS(x, ratio=0.5)
    fps_center, x, fps_idx = my_fps.get()
