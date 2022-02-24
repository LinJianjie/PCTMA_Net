import open3d as o3d
import os
import sys
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
#from components.dataSet import ModelNet
from components.dataUtils import *


class PlotModelNet40:
    def __int__(self, model):
        self.data = model.data
        self.label = model.category_label
        self.shape_name = model.shape_name

        self.pcd = o3d.geometry.PointCloud()

    def __get_data_with_label(self, label):
        pcd = o3d.geometry.PointCloud()
        result = np.where(self.label == self.shape_name[label])[0]
        if result is not None:
            return result, o3d.utility.Vector3dVector(self.data[result, :, :])

    def draw(self, data):
        self.pcd.points = data
        o3d.visualization.draw_geometries([self.pcd])


def draw_function(pcd):
    # if isinstance(pcd, o3d.open3d_pybind.geometry.PointCloud):
    pcd.paint_uniform_color([0, 0, 1])
    o3d.visualization.draw_geometries([pcd])


def toOpen3D(x):
    # if isinstance(x, np.asarray):
    #     pcd = o3d.geometry.PointCloud()
    #     pcd.points = o3d.utility.Vector3dVector(x)
    #     return pcd
    # else:
    #     raise Exception("x is not numpy")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(x)
    return pcd


if __name__ == '__main__':
    model = ModelNet(num_points=128)
    airplane_pc = model.get_with_shapeName("bathtub")
    reduced_airplane_pc = model.get_reduced_points(airplane_pc[1], 128)
    poin = toOpen3D(reduced_airplane_pc)
    #draw_function(poin)
    # airplane_pc = model.get_with_shapeName("guitar")
    # reduced_airplane_pc = model.get_reduced_points(airplane_pc[0], 1024)
    # partial_reduced = model.get_partial_points(airplane_pc[0])
    # # reduced_airplane_pc = reduced_airplane_pc + np.asarray[0, 1, 0]
    # reduced_airplane_pcd = toOpen3D(reduced_airplane_pc)
    # draw_function(reduced_airplane_pcd)
    # partial_pcd = toOpen3D(partial_reduced)
    # draw_function(partial_pcd)
    # pc = torch.from_numpy(airplane_pc[0]).unsqueeze(dim=0)
    # pcd_ori = o3d.geometry.PointCloud()
    # pcd_ori.points = o3d.utility.Vector3dVector(airplane_pc[0])
    # pt = DataUtils.normalizedPC(pc)
    # pt = pt.squeeze().numpy()
    # pcd_nor = o3d.geometry.PointCloud()
    # pcd_nor.points = o3d.utility.Vector3dVector(pt)

    # print(pt.shape)
    # y = np.floor(pt / np.array([0.1, 0.1, 0.1]))
    # y1, index, inverse, acount = np.unique(y, axis=0, return_index=True, return_inverse=True, return_counts=True)
    # print("index ")
    # at = 1
    # print(y1)
    # print(y1[at])
    # print(index.shape)
    # print(y[index[at]])
    # # print(acount)
    # print("inverse")
    # print(y[0])
    # print(inverse[0:10])
    # print(y1[inverse[0]])
    # t = np.where(inverse == 1)
    # print(acount[1])
    # print("t: ", t)
    # print("tttt")
    # print(y1[inverse[t]])
