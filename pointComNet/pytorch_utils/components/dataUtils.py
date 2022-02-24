import os
import sys
import open3d as o3d

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
import numpy as np
import torch
import torch.nn.parallel
import torch


# import torch_geometric.nn
# from torch_cluster import fps
# from components.externs_tools.fps import fps


class DataUtils:
    @staticmethod
    def normalizedPC(points, maxrlist=None, mean=None):
        assert points.shape[2] == 3
        assert points.dim() == 3
        if mean is None:
            points, mean = DataUtils.zeroCenter(points)
        else:
            points, mean = DataUtils.zeroCenter(points, mean)

        if maxrlist is None:
            rlist = torch.norm(points, dim=2, p=2, keepdim=True)
            maxrlist = torch.max(rlist, dim=1, keepdim=True)[0]
            maxrlist = maxrlist.repeat(1, points.shape[1], points.shape[2])
        points_ = points / maxrlist
        return points_, maxrlist, mean
        # return (points_)

    @staticmethod
    def zeroCenter(points, mean=None):  # for B*N*3
        assert points.shape[2] == 3
        assert points.dim() == 3
        if mean is None:
            mean = points.mean(1).reshape(points.shape[0], 1, points.shape[2])
            points_ = points - mean
        else:
            points_ = points - mean
        return points_, mean

    @staticmethod
    def angle_axis(angle, axis):
        # type: (float, np.ndarray) -> float
        r"""Returns a 4x4 rotation matrix that performs a rotation around axis by angle

        Parameters
        ----------
        angle : float
            Angle to rotate by
        axis: np.ndarray
            Axis to rotate about

        Returns
        -------
        torch.Tensor
            3x3 rotation matrix
        """
        u = axis / np.linalg.norm(axis)
        cosval, sinval = np.cos(angle), np.sin(angle)

        # yapf: disable
        cross_prod_mat = np.array([[0.0, -u[2], u[1]],
                                   [u[2], 0.0, -u[0]],
                                   [-u[1], u[0], 0.0]])

        R = torch.from_numpy(
            cosval * np.eye(3)
            + sinval * cross_prod_mat
            + (1.0 - cosval) * np.outer(u, u)
        )
        # yapf: enable
        return R.float()

    # @staticmethod
    # def farthest_point_sample(xyz, npoint):
    #     """
    #     Input:
    #         xyz: pointcloud data, [B, N, 3]
    #         npoint: number of samples
    #     Return:
    #         centroids: sampled pointcloud index, [B, npoint]
    #     """
    #     device = xyz.device
    #     B, N, C = xyz.shape
    #     centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    #     distance = torch.ones(B, N).to(device) * 1e10
    #     farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    #     print(farthest)
    #     batch_indices = torch.arange(B, dtype=torch.long).to(device)
    #     for i in range(npoint):
    #         centroids[:, i] = farthest
    #         centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
    #         dist = torch.sum((xyz - centroid) ** 2, -1)
    #         mask = dist < distance
    #         distance[mask] = dist[mask]
    #         farthest = torch.max(distance, -1)[1]
    #     return centroids

    # @staticmethod
    # def fps(x, ratio=0.5, random_start=True):
    #     pointCloudPose = torch.from_numpy(x)
    #     pose_batch = pointCloudPose
    #     batch_index = torch.arange(1)
    #     batch_fps = batch_index.repeat(x.shape[0], 1).transpose(0, 1).reshape(-1)
    #     fps_idx = fps(pose_batch, batch_fps, ratio, random_start=random_start)
    #     point_cloud = pose_batch[fps_idx].reshape(-1, 3)
    #     return point_cloud, fps_idx
    @staticmethod
    def unitSpher(points):
        norm = torch.norm(points, dim=2, p=2, keepdim=True)
        points_ = points / norm
        return points_

    @staticmethod
    def scalingPoints(points):
        rlist = torch.norm(points, dim=2, p=2, keepdim=True)
        maxrlist = torch.max(rlist, dim=1, keepdim=True)[0]
        maxrlist = maxrlist.repeat(1, points.shape[1], 1)
        points_ = points / maxrlist
        return points_

    @staticmethod
    def singleScalingPoint(point):
        rlist = torch.norm(point, dim=1, p=2, keepdim=True)
        maxrlist = torch.min(rlist, dim=0, keepdim=True)[0]
        maxrlist = maxrlist.repeat(1, point.shape[1])
        points_ = point / maxrlist
        return points_

    @staticmethod
    def square_distance(src, dst):
        """
        Calculate Euclid distance between each two points.

        src^T * dst = xn * xm + yn * ym + zn * zm；
        sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
        sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
        dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
             = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

        Input:
            src: source points, [B, N, C]
            dst: target points, [B, M, C]
        Output:
            dist: per-point square distance, [B, N, M]
        """
        B, N, _ = src.shape
        _, M, _ = dst.shape
        dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
        dist += torch.sum(src ** 2, -1).view(B, N, 1)
        dist += torch.sum(dst ** 2, -1).view(B, 1, M)
        return dist

    @staticmethod
    def index_points(points, idx):
        """

        Input:
            points: input points data, [B, N, C]
            idx: sample index data, [B, S]
        Return:
            new_points:, indexed points data, [B, S, C]
        """
        device = points.device
        B = points.shape[0]
        view_shape = list(idx.shape)
        view_shape[1:] = [1] * (len(view_shape) - 1)
        repeat_shape = list(idx.shape)
        repeat_shape[0] = 1
        batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
        new_points = points[batch_indices, idx, :]
        return new_points


class PointCloudBase(object):
    def toTensor(self, points):
        return torch.from_numpy(points).float()

    def __call__(self, points):
        return points


class PointCloudScale(PointCloudBase):
    def __init__(self, lo=0.8, hi=1.25):
        self.lo, self.hi = lo, hi

    def __call__(self, points):
        scale = np.random.uniform(self.lo, self.hi)
        points[:, 0:3] *= scale
        return points


class PointCloudAzimuthalRotations(PointCloudBase):
    def __init__(self, axis=np.array([0.0, 0.0, 1.0])):
        self.axis = axis

    def get_rot(self):
        rotation_angle = np.random.uniform() * 2 * np.pi
        rotation_matrix = DataUtils.angle_axis(rotation_angle, self.axis)
        return rotation_matrix

    def __call__(self, points):
        rotation_angle = np.random.uniform() * 2 * np.pi
        rotation_matrix = DataUtils.angle_axis(rotation_angle, self.axis)

        normals = points.size(1) > 3
        if not normals:
            return torch.matmul(points, rotation_matrix.t())
        else:
            pc_xyz = points[:, 0:3]
            pc_normals = points[:, 3:]
            points[:, 0:3] = torch.matmul(pc_xyz, rotation_matrix.t())
            points[:, 3:] = torch.matmul(pc_normals, rotation_matrix.t())
            return points


class PointCloudArbitraryRotation(PointCloudBase):
    def _get_angles(self):
        angles = np.random.uniform(size=3) * 2 * np.pi
        return angles

    def get_rot(self):
        angles = self._get_angles()
        Rx = DataUtils.angle_axis(angles[0], np.array([1.0, 0.0, 0.0]))
        Ry = DataUtils.angle_axis(angles[1], np.array([0.0, 1.0, 0.0]))
        Rz = DataUtils.angle_axis(angles[2], np.array([0.0, 0.0, 1.0]))
        rotation_matrix = torch.matmul(torch.matmul(Rz, Ry), Rx)
        return rotation_matrix

    def __call__(self, points):
        angles = self._get_angles()
        Rx = DataUtils.angle_axis(angles[0], np.array([1.0, 0.0, 0.0]))
        Ry = DataUtils.angle_axis(angles[1], np.array([0.0, 1.0, 0.0]))
        Rz = DataUtils.angle_axis(angles[2], np.array([0.0, 0.0, 1.0]))
        rotation_matrix = torch.matmul(torch.matmul(Rz, Ry), Rx)
        normals = points.size(1) > 3
        if not normals:
            return torch.matmul(points, rotation_matrix.t())
        else:
            pc_xyz = points[:, 0:3]
            pc_normals = points[:, 3:]
            points[:, 0:3] = torch.matmul(pc_xyz, rotation_matrix.t())
            points[:, 3:] = torch.matmul(pc_normals, rotation_matrix.t())
            return points


class PointCloudRotatePerturbation(PointCloudBase):
    def __init__(self, angle_sigma=0.06, angle_clip=0.18):
        self.angle_sigma, self.angle_clip = angle_sigma, angle_clip

    def _get_angles(self):
        angles = np.clip(self.angle_sigma * np.random.randn(3), -self.angle_clip, self.angle_clip)
        return angles

    def __call__(self, points):
        angles = self._get_angles()
        Rx = DataUtils.angle_axis(angles[0], np.array([1.0, 0.0, 0.0]))
        Ry = DataUtils.angle_axis(angles[1], np.array([0.0, 1.0, 0.0]))
        Rz = DataUtils.angle_axis(angles[2], np.array([0.0, 0.0, 1.0]))

        rotation_matrix = torch.matmul(torch.matmul(Rz, Ry), Rx)

        normals = points.size(1) > 3
        if not normals:
            return torch.matmul(points, rotation_matrix.t())
        else:
            pc_xyz = points[:, 0:3]
            pc_normals = points[:, 3:]
            points[:, 0:3] = torch.matmul(pc_xyz, rotation_matrix.t())
            points[:, 3:] = torch.matmul(pc_normals, rotation_matrix.t())
            return points


class PointCloudJitter(PointCloudBase):
    def __init__(self, std=0.002, clip=0.05):
        self.std, self.clip = std, clip

    def __call__(self, points):
        jittered_data = (points.new(points.size(0), 3).normal_(mean=0.0, std=self.std).clamp_(-self.std, self.std))
        points[:, 0:3] += jittered_data
        return points


class PointCloudTranslate(PointCloudBase):
    def __init__(self, translate_range=0.1):
        self.translate_range = translate_range

    def __call__(self, points):
        translation = np.random.uniform(-self.translate_range, self.translate_range)
        points[:, 0:3] += translation
        return points
