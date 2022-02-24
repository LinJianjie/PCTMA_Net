import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
import torch.nn.parallel
import torch_geometric.nn
from components.dataUtils import *
from components.dataSet import ModelNet
from components.dataUtils import *
from components.ioUtils import *


class ClusterKNN:
    def __init__(self, points, K, includeSelf=False):
        self.K = K
        self.edge_index_knn = torch_geometric.nn.knn_graph(points, k=K, loop=includeSelf)

    def query(self, index):
        return self.edge_index_knn[:, self.edge_index_knn[1, :] == index]


class BatchKnn:  # B,N,D
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
            assign_index = torch_geometric.nn.knn(self.pointCRange, queryPointsCRange, self.maxKnn, batch, batch_query)
        else:
            assign_index = torch_geometric.nn.knn(self.pointCRange, self.pointCRange, self.maxKnn, batch, batch)
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


class PCAAlignment:
    def pcaCoordinate(self, coord, choose):
        coordList = []
        coord0 = coord.clone().detach()
        coord1 = coord.clone().detach()
        coord1[:, :, 0] = coord[:, :, 0] * -1
        coord1[:, :, 1] = coord[:, :, 1] * -1
        coord1[:, :, 2] = coord[:, :, 2] * -1

        coord2 = coord.clone().detach()
        coord2[:, :, 0] = coord[:, :, 0] * -1
        coord2[:, :, 1] = coord[:, :, 1] * 1
        coord2[:, :, 2] = coord[:, :, 2] * 1

        coord3 = coord.clone().detach()
        coord3[:, :, 0] = coord[:, :, 0] * 1
        coord3[:, :, 1] = coord[:, :, 1] * -1
        coord3[:, :, 2] = coord[:, :, 2] * 1

        coord4 = coord.clone().detach()
        coord4[:, :, 0] = coord[:, :, 0] * 1
        coord4[:, :, 1] = coord[:, :, 1] * 1
        coord4[:, :, 2] = coord[:, :, 2] * -1
        #
        coord5 = coord.clone().detach()
        coord5[:, :, 0] = coord[:, :, 0] * -1
        coord5[:, :, 1] = coord[:, :, 1] * -1
        coord5[:, :, 2] = coord[:, :, 2] * 1

        coord6 = coord.clone().detach()
        coord6[:, :, 0] = coord[:, :, 0] * -1
        coord6[:, :, 1] = coord[:, :, 1] * 1
        coord6[:, :, 2] = coord[:, :, 2] * -1

        coord7 = coord.clone().detach()
        coord7[:, :, 0] = coord[:, :, 0] * 1
        coord7[:, :, 1] = coord[:, :, 1] * -1
        coord7[:, :, 2] = coord[:, :, 2] * -1
        if choose >= 1:
            coordList.append(coord0)
        elif choose >= 2:
            coordList.append(coord1)
        elif choose >= 3:
            coordList.append(coord2)
        elif choose >= 4:
            coordList.append(coord3)
        elif choose >= 5:
            coordList.append(coord4)
        elif choose >= 6:
            coordList.append(coord5)
        elif choose >= 7:
            coordList.append(coord6)
        elif choose >= 8:
            coordList.append(coord7)
        else:
            print("the choose value is an error")

        return coordList

    def alignment(self, points, type="train", choose=1):
        pca = torch.pca_lowrank(points)
        pcaCoordinate = pca[2]
        if type == "train":
            new_coordinate = self.pcaCoordinate(pcaCoordinate, choose)
            new_points = []
            for coordinate in new_coordinate:
                new_points.append(torch.bmm(points, coordinate))
        else:
            new_points = torch.bmm(points, pcaCoordinate)


class PointVoxelization(object):
    def __init__(self, point, leaf_x):
        self.points = point
        self.leaf_x = leaf_x
        self.leaf = np.array([self.leaf_x, self.leaf_x, self.leaf_x])
        self.r = int(1 / self.leaf_x)
        self.voxel_grid = np.zeros([self.r, self.r, self.r])
        self.unique_voxel_index = None
        self.unique_pt_to_voxel_index = None
        self.unique_voxel_to_pt_index = None
        self.unique_voxel_account = None

    def assign_to_voxel(self):
        center_voxel = self.center_Voxel()
        for i, index in enumerate(self.unique_voxel_index):
            index = index.astype(int)
            self.voxel_grid[index[0], index[1], index[2]] = 1
        # index = self.unique_voxel_index.astype(int)
        # self.voxel_grid[index] = 1
        print(self.voxel_grid)

    def voxelization(self):
        shift_pc = DataUtils.zeroCenter(self.points)
        normalized_pc = DataUtils.normalizedPC(shift_pc)
        return torch.floor(normalized_pc / self.leaf)

    def voxelization_numpy(self):
        normalized_pc = self.points
        voxel_index = np.floor(normalized_pc / np.array(self.leaf))
        voxel_index = np.clip(voxel_index, 0, self.r - 1)
        self.unique_voxel_index, self.unique_pt_to_voxel_index, self.unique_voxel_to_pt_index, self.unique_voxel_account = np.unique(
            voxel_index,
            axis=0,
            return_index=True,
            return_inverse=True,
            return_counts=True)

    def get_points_at_voxel(self, at_voxel):
        at_voxel = np.array(at_voxel)
        point = [self.points[np.where(self.unique_voxel_to_pt_index == at_voxel[i])] for i in range(at_voxel.shape[0])]
        return point

    def center_Voxel(self):
        at_voxel = np.arange(self.unique_voxel_index.shape[0])
        point = self.get_points_at_voxel(at_voxel)
        center_voxel = [np.mean(point[i], axis=0) for i in range(len(point))]
        return np.asarray(center_voxel)

    def get_voxel_at_point(self, at_point):
        voxel_index_at_point = self.unique_voxel_index[self.unique_voxel_to_pt_index[at_point]]
        return voxel_index_at_point

    def get_len_voxel(self):
        return self.unique_voxel_index.shape[0]

    def __getitem__(self, item):
        return self.unique_voxel_index[item]

    def get_Neighbor(self, max_knn):
        assign_index = torch_geometric.nn.knn(torch.from_numpy(self.unique_voxel_index),
                                              torch.from_numpy(self.unique_voxel_index), max_knn)
        nn = self.unique_voxel_index[assign_index[1, :], :]
        return nn.reshape(self.unique_voxel_index.shape[0], max_knn, self.unique_voxel_index.shape[1]), \
               assign_index[1, :].reshape(self.unique_voxel_index.shape[0], max_knn)

    def get_Neighbor_in_Voxel(self, max_knn):
        point_index = torch.arange(0, self.points.shape[0])
        voxel_index_at_point_value = self.unique_voxel_index[self.unique_voxel_to_pt_index[point_index]]
        voxel_index_at_point = self.unique_voxel_to_pt_index[point_index]
        voxel_knn, voxel_knn_index = self.get_Neighbor(max_knn)
        # voxel_knn_based_knn = voxel_knn[voxel_index_at_point, :, :]
        point_nn = [self.get_points_at_voxel(voxel_knn_index[i, :].numpy()) for i in range(voxel_knn_index.shape[0])]
        return point_nn


if __name__ == '__main__':
    # x = torch.rand(4, 1024, 3)
    # pv = PointVoxelization(x, 0.1, 0.1, 0.1)
    # # y = pv.voxelization()
    # pt = torch.rand(1, 1024, 3)
    # # pt = DataUtils.zeroCenter(pt)
    # pt = DataUtils.normalizedPC(pt)
    # pt = pt.squeeze(dim=0)
    # pt = pt.numpy()
    # y = np.floor(pt / np.array([0.1, 0.1, 0.1]))
    # print(y)
    # y1 = np.unique(y, axis=0)
    # print("y1: ")
    # print(y1.shape)

    model = ModelNet(num_points=1024)
    airplane_pc = model.get_with_shapeName("guitar")
    # pcd_before = o3d.geometry.PointCloud()
    # pcd_before.points = o3d.utility.Vector3dVector(airplane_pc[0])
    # pcd_before.paint_uniform_color([0.5, 0, 0.5])
    # pc = torch.from_numpy(airplane_pc[0]).unsqueeze(dim=0)
    # pc = model.rot(pc)
    # pcd_ori = o3d.geometry.PointCloud()
    # pcd_ori.points = o3d.utility.Vector3dVector(pc.squeeze().numpy())
    # pcd_ori.paint_uniform_color([1, 0, 1])
    reduced_pc_1024_org = model.get_reduced_points(airplane_pc[20], 1024)
    jitter = PointCloudJitter(std=0.002)
    reduced_pc_1024_jitter_0002 = (reduced_pc_1024_org.new(reduced_pc_1024_org.size(0), 3).normal_(mean=0.0,
                                                                                                   std=0.002)) + reduced_pc_1024_org + torch.tensor(
        [0, 0, -0.75])

    reduced_pc_1024_jitter_001 = (reduced_pc_1024_org.new(reduced_pc_1024_org.size(0), 3).normal_(mean=0.0,
                                                                                                  std=0.01)) + reduced_pc_1024_org + torch.tensor(
        [0, 0, -1.5])
    # np.savetxt("noise_0002.dat", reduced_pc_1024_jitter_0002.numpy())
    pcd_reduced_1024_org = o3d.geometry.PointCloud()
    pcd_reduced_1024_org.points = o3d.utility.Vector3dVector(reduced_pc_1024_org.numpy())
    pcd_reduced_1024_org.paint_uniform_color([0, 0, 1])

    pcd_reduced_1024_jitter_0002 = o3d.geometry.PointCloud()
    pcd_reduced_1024_jitter_0002.points = o3d.utility.Vector3dVector(reduced_pc_1024_jitter_0002.numpy())
    pcd_reduced_1024_jitter_0002.paint_uniform_color([0, 0, 1])

    pcd_reduced_1024_jitter_001 = o3d.geometry.PointCloud()
    pcd_reduced_1024_jitter_001.points = o3d.utility.Vector3dVector(reduced_pc_1024_jitter_001.numpy())
    pcd_reduced_1024_jitter_001.paint_uniform_color([0, 0, 1])

    # reduced_pc_512 = model.get_reduced_points(airplane_pc[0], 512)
    # reduced_pc_512 = reduced_pc_512 + torch.tensor([0, 0, 1])
    # pcd_reduced_512 = o3d.geometry.PointCloud()
    # pcd_reduced_512.points = o3d.utility.Vector3dVector(reduced_pc_512.numpy())
    # pcd_reduced_512.paint_uniform_color([0, 0, 1])
    #
    # reduced_pc_256 = model.get_reduced_points(airplane_pc[0], 256)
    # reduced_pc_256 = reduced_pc_256 + torch.tensor([0, 0, 2])
    # pcd_reduced_256 = o3d.geometry.PointCloud()
    # pcd_reduced_256.points = o3d.utility.Vector3dVector(reduced_pc_256.numpy())
    # pcd_reduced_256.paint_uniform_color([0, 0, 1])

    # pt = DataUtils.normalizedPC(pc)
    # pt = pt.squeeze().numpy()
    # pcd_nor = o3d.geometry.PointCloud()
    # pcd_nor.points = o3d.utility.Vector3dVector(pt)
    # pcd_nor.paint_uniform_color([1, 0, 1])
    # mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=5)
    # mesh_sphere.paint_uniform_color([0.1, 0.1, 0.7])
    o3d.visualization.draw_geometries(
        [pcd_reduced_1024_org, pcd_reduced_1024_jitter_0002, pcd_reduced_1024_jitter_001])
    # pv = PointVoxelization(pt, 0.01)
    # pv.voxelization_numpy()
    # nn = pv.get_Neighbor(5)
    # pv.get_Neighbor_in_Voxel(5)
    # center_voxel = pv.center_Voxel()
    # pv.assign_to_voxel()
