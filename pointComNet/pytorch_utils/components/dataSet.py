import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
import glob
import h5py
from torch.utils.data import Dataset
from components.dataUtils import *
from components.torch_cluster_sampling import *
from components.constant import *
import numpy as np
from components.ioUtils import readText
import torch
import itertools
from torch.utils.data import DataLoader
import transforms3d
from components.dataLoader import Completion3Ddownload, ModelNet40download, KITTIDownload, ShapeNetDownload

torch.multiprocessing.set_sharing_strategy('file_system')


def augment_cloud(Ps, pc_augm_rot=True, pc_augm_mirror_prob=0.5):
    """" Augmentation on XYZ and jittering of everything """
    M = transforms3d.zooms.zfdir2mat(1)
    if pc_augm_rot:
        angle = np.random.uniform(0, 2 * np.pi)
        M = np.dot(transforms3d.axangles.axangle2mat([0, 1, 0], angle), M)  # y=upright assumption
    if pc_augm_mirror_prob > 0:  # mirroring x&z, not y
        if np.random.random() < pc_augm_mirror_prob / 2:
            M = np.dot(transforms3d.zooms.zfdir2mat(-1, [1, 0, 0]), M)
        if np.random.random() < pc_augm_mirror_prob / 2:
            M = np.dot(transforms3d.zooms.zfdir2mat(-1, [0, 0, 1]), M)
    result = []
    for P in Ps:
        P[:, :3] = np.dot(P[:, :3], M.T)
        result.append(P)
    return result


def pad_cloudN(P, Nin):
    """ Pad or subsample 3D Point cloud to Nin number of points """
    N = P.shape[0]
    P = P[:].astype(np.float32)

    rs = np.random.random.__self__
    choice = np.arange(N)
    if N > Nin:  # need to subsample
        ii = rs.choice(N, Nin)
        choice = ii
    elif N < Nin:  # need to pad by duplication
        ii = rs.choice(N, Nin - N)
        choice = np.concatenate([range(N), ii])
    P = P[choice, :]

    return P


def differentResolution(P, Nin):
    p1 = pad_cloudN(P, Nin)
    p2 = pad_cloudN(p1, 2048)
    return p2


class PointCompletionShapeNet(Dataset):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    foldName = "dataset2019/shapenet"
    DATA_DIR = os.path.join(BASE_DIR, dataFile, foldName)

    def __init__(self, num_points, partition='train', transformationType="Z", pertubation=False,
                 use_fps=True, class_choice=None, has_augment_cloud=False, has_down_sampling=False, use_kitti=False):
        Completion3Ddownload()
        self.synsetoffset2category = os.path.join(self.DATA_DIR, "synsetoffset2category.txt")
        self.test = os.path.join(self.DATA_DIR, "test.list")
        self.train = os.path.join(self.DATA_DIR, "train.list")
        self.val = os.path.join(self.DATA_DIR, "val.list")
        self.num_points = num_points
        self.category_label = dict()
        self.category = []
        self.use_fps = use_fps
        self.use_kitti = use_kitti
        self.use_perturbation = pertubation
        if transformationType == "Z":
            self.rot = PointCloudAzimuthalRotations()
        elif transformationType == "S":
            self.rot = PointCloudArbitraryRotation()
        else:
            self.rot = PointCloudBase()
        if self.use_perturbation:
            self.perturbation = PointCloudJitter()
        self.gt_data_, self.partial_data_, self.label_ = self.load(partition=partition)
        # TODO doing the preprocessing before __getitem__

        # self.gt_data_ = self.down_sampling(self.gt_data_)
        # self.gt_data_ = self.rotate(self.gt_data_)

        # self.partial_data_ = self.down_sampling(self.partial_data_)
        # self.partial_data_ = self.rotate(self.partial_data_)
        if class_choice is not None:
            print("class choice")
            self.gt_data_, self.partial_data_, self.label_ = self.get_with_shapeNames(names=class_choice)
        # self.gt_data_, maxrlist, mean, = DataUtils.normalizedPC(torch.from_numpy(self.gt_data_))
        # self.partial_data_, maxrlist, _ = DataUtils.normalizedPC(torch.from_numpy(self.partial_data_),
        #                                                         maxrlist=maxrlist, mean=mean)
        if has_augment_cloud and partition == "train":
            print("===>start to augment_cloud")
            gt_data_list = []
            partial_list_ = []
            for i in range(self.gt_data_.shape[0]):
                temp_gt, temp_partial = augment_cloud([self.gt_data_[i], self.partial_data_[i]])
                gt_data_list.append(temp_gt)
                partial_list_.append(temp_partial)
            # self.gt_data_, self.partial_data_ = augment_cloud([self.gt_data_[i], self.partial_data_[i]])
            self.gt_data_ = np.asarray(gt_data_list)
            self.partial_data_ = np.asarray(partial_list_)
            print("gt_data_: ", self.gt_data_.shape)
            print("partial_data_: ", self.partial_data_.shape)
            print("==> finished to augment_cloud")
        if has_down_sampling and partition == "val":
            print("star to random sampling")
            gt_data_list = []
            partial_list_ = []
            down_sampling_npts = 1024
            print("the downsampling: ", down_sampling_npts)
            for i in range(self.partial_data_.shape[0]):
                temp_partial = differentResolution(self.partial_data_[i], down_sampling_npts)
                partial_list_.append(temp_partial)
            self.partial_data_ = np.asarray(partial_list_)
            print("gt_data_: ", self.gt_data_.shape)
            print("partial_data_: ", self.partial_data_.shape)
            print("==> finished to down_sampling")

    def __getitem__(self, item):
        if self.use_fps is False:
            pt_idxes = np.arange(0, self.gt_data_[item].shape[0])
            np.random.shuffle(pt_idxes)
            gt_point_cloud = self.gt_data_[item][pt_idxes[:self.num_points]].copy()
            pt_idxes_partial = np.arange(0, self.partial_data_[item].shape[0])
            np.random.shuffle(pt_idxes_partial)
            partial_point_cloud = self.partial_data_[item][pt_idxes_partial[:self.num_points]].copy()
        else:
            # _, gt_point_cloud, fps_index = farthest_point_sampling(torch.from_numpy(self.gt_data_[item]),
            #                                                       ratio=self.num_points / self.gt_data_[item].shape[0])
            # _, partial_point_cloud, fps_index = farthest_point_sampling(torch.from_numpy(self.partial_data_[item]),
            #                                                            ratio=self.num_points /
            #                                                                  self.partial_data_[item].shape[0])
            if self.gt_data_ is not None:
                gt_point_cloud = self.gt_data_[item]
            else:
                gt_point_cloud = None
            partial_point_cloud = self.partial_data_[item]
        if self.label_ is not None:
            label_point_cloud = self.label_[item]
        else:
            label_point_cloud = None
        # if gt_point_cloud is not None:
        #     gt_point_cloud = self.rot(gt_point_cloud)
        # else:
        #     gt_point_cloud = None
        # partial_point_cloud = self.rot(partial_point_cloud)
        return gt_point_cloud, partial_point_cloud, label_point_cloud

    @staticmethod
    def evaluation_class(label_name):
        class_name = ["Plane", "Cabinet", "Car", "Chair", "Lamp", "Couch", "Table", "Watercraft"]
        for i, name in enumerate(class_name):
            if name.lower() == label_name.lower():
                return name

    def down_sampling(self, data):
        pt_all = []
        for i in range(data.shape[0]):
            _, down_sampling_pt, fps_index = farthest_point_sampling(torch.from_numpy(data[i]),
                                                                     ratio=self.num_points / data[i].shape[0])
            pt_all.append(down_sampling_pt)
        pt_all = torch.from_numpy(np.asarray(pt_all))
        return pt_all

    def rotate(self, data):
        pt_all = []
        for i in range(data.shape[0]):
            pc = self.rot(data[i])
            pt_all.append(pc)
        pt_all = torch.from_numpy(np.asarray(pt_all))
        return pt_all

    def __len__(self):
        if self.gt_data_ is not None:
            return self.gt_data_.shape[0]
        else:
            return self.partial_data_.shape[0]

    def load(self, partition='train'):
        # PointCompletionShapeNet.download()
        self.category = self.get_category_file()
        if partition == "train":
            print("----load train dataset")
            gt, partial, label = self.__load_train()
            print(".... finished load train dataset")
            return gt, partial, label
        elif partition == "test":
            print("----load test dataset")
            gt, partial, label = self.__load_test()
            return gt, partial, label
        elif partition == "val":
            print("----load val dataset")
            if self.use_kitti:
                print("----load val kitti dataset")
                gt, partial, label = self.__load_kitti_val()
            else:
                print("----load val completion3D dataset")
                gt, partial, label = self.__load_val()
            return gt, partial, label
        else:
            raise Exception("no partition is found")

    def __load_train(self):
        train_data_file = self.get_train_file()
        gt_train_data = []
        partial_train_data = []
        train_label = []
        i = 0
        for file in train_data_file:
            h5_name_gt = os.path.join(self.DATA_DIR, 'train/gt', file[0], '%s.h5' % file[1])
            # for h5_name in glob.glob(os.path.join(self.DATA_DIR, 'train/gt', file[0], '%s.h5' % file[1])):
            f1 = h5py.File(h5_name_gt, 'r')
            i = i + 1
            data = f1['data'][:].astype('float32')
            f1.close()
            gt_train_data.append(data)
            # for h5_name in glob.glob(os.path.join(self.DATA_DIR, 'train/partial', file[0], '%s.h5' % file[1])):
            h5_name_partial = os.path.join(self.DATA_DIR, 'train/partial', file[0], '%s.h5' % file[1])
            f2 = h5py.File(h5_name_partial, 'r')
            data = f2['data'][:].astype('float32')
            f2.close()
            partial_train_data.append(data)
            train_label.append(self.category_label[file[0]])
        # print(train_label)

        gt_train_data = np.stack(gt_train_data, axis=0)
        partial_train_data = np.stack(partial_train_data, axis=0)
        train_label = np.stack(train_label, axis=0)
        print("gt: ", gt_train_data.shape)
        return gt_train_data, partial_train_data, train_label

    def __load_test(self):
        test_data_file = self.get_test_file()
        partial_test_data = []
        i = 0
        for file in test_data_file:
            for h5_name in glob.glob(os.path.join(self.DATA_DIR, 'test/partial', file[0], '%s.h5' % file[1])):
                f = h5py.File(h5_name, 'r')
                data = f['data'][:].astype('float32')
                f.close()
                partial_test_data.append(data)
        partial_test_data = np.stack(partial_test_data, axis=0)
        label = torch.ones(partial_test_data.shape[0])
        return partial_test_data, partial_test_data, label

    def __load_kitti_val(self):
        h5_name = os.path.join(self.DATA_DIR, 'val/kitti_data.h5')
        f = h5py.File(h5_name, 'r')
        data = f["car"][:].astype('float32')
        partial_val_data = data
        label = np.ones(partial_val_data.shape[0])
        return partial_val_data, partial_val_data, label

    def __load_val(self):
        val_data_file = self.get_val_file()
        gt_val_data = []
        partial_val_data = []
        val_label = []
        for file in val_data_file:
            for h5_name in glob.glob(os.path.join(self.DATA_DIR, 'val/gt', file[0], '%s.h5' % file[1])):
                f = h5py.File(h5_name, 'r')
                data = f['data'][:].astype('float32')
                f.close()
                gt_val_data.append(data)
            for h5_name in glob.glob(os.path.join(self.DATA_DIR, 'val/partial', file[0], '%s.h5' % file[1])):
                f = h5py.File(h5_name, 'r')
                data = f['data'][:].astype('float32')
                f.close()
                partial_val_data.append(data)
            val_label.append(self.category_label[(file[0])])

        gt_val_data = np.stack(gt_val_data, axis=0)
        partial_val_data = np.stack(partial_val_data, axis=0)
        val_label = np.stack(val_label, axis=0)
        return gt_val_data, partial_val_data, val_label

    def get_category_file(self):
        category = dict()
        data = readText(self.synsetoffset2category, '\t')
        for i, x in enumerate(data):
            category[x[0]] = int(x[1])
            self.category_label[x[1]] = i
        return category

    def label_to_category(self, label):
        key_name = None
        for key, item in self.category_label.items():  # for name, age in dictionary.iteritems():  (for Python 2.x)
            if item == label:
                key_name = key
                break
        for key, item in self.category.items():
            if item == int(key_name):
                return key
        # name = self.category[key_name]

    def get_with_shapeName(self, name):
        result = np.where(self.label_ == self.category_label[self.category[name]])[0]
        labels = list(itertools.repeat(self.category[name], len(result)))
        return result, labels

    def get_with_shapeNames(self, names):
        gt_ = []
        partial_ = []
        labels_ = []
        for name in names:
            result, label_ = self.get_with_shapeName(name.lower())
            gt_.append(self.gt_data_[result, :, :])
            partial_.append(self.partial_data_[result, :, :])
            labels_.append(label_)
        gt_ = np.concatenate(gt_, axis=0)
        partial_ = np.concatenate(partial_, axis=0)
        labels_ = np.concatenate(labels_, axis=0)
        return gt_, partial_, labels_

    def get_train_file(self):
        train_data = readText(self.train, '/')
        return train_data

    def get_test_file(self):
        test_data = readText(self.test, '/')
        return test_data

    def get_val_file(self):
        val_data = readText(self.val, '/')
        return val_data

    # @staticmethod
    # def download():
    #     # TODO
    #     BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    #     DATA_DIR = os.path.join(BASE_DIR, dataFile)
    #     if not os.path.exists(DATA_DIR):
    #         os.makedirs(DATA_DIR, exist_ok=True)
    #     if not os.path.exists(os.path.join(DATA_DIR, 'shapenet')):
    #         www = 'http://download.cs.stanford.edu/downloads/completion3d/dataset2019.zip'
    #         zipfile = os.path.basename(www)
    #         os.system('wget --no-check-certificate %s; unzip %s' % (www, zipfile))
    #         os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
    #         os.system('rm %s' % zipfile)
    #
    # @staticmethod
    # def download16K():
    #     BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    #     DATA_DIR = os.path.join(BASE_DIR, dataFile, "PointCompletion")
    #     if not os.path.exists(DATA_DIR):
    #         os.makedirs(DATA_DIR, exist_ok=True)
    #     if not os.path.exists(os.path.join(DATA_DIR, 'shapenet')):
    #         www = 'http:// download.cs.stanford.edu/downloads/completion3d/shapenet16K2019.zip'
    #         zipfile = os.path.basename(www)
    #         os.system('wget --no-check-certificate %s; unzip %s' % (www, zipfile))
    #         os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
    #         os.system('rm %s' % zipfile)


class ModelNet(Dataset):
    shape_name = {
        'airplane': 0,
        'bathtub': 1,
        'bed': 2,
        'bench': 3,
        'bookshelf': 4,
        'bottle': 5,
        'bowl': 6,
        'car': 7,
        'chair': 8,
        'cone': 9,
        'cup': 10,
        'curtain': 11,
        'desk': 12,
        'door': 13,
        'dresser': 14,
        'flower_pot': 15,
        'glass_box': 16,
        'guitar': 17,
        'keyboard': 18,
        'lamp': 19,
        'laptop': 20,
        'mantel': 21,
        'monitor': 22,
        'night_stand': 23,
        'person': 24,
        'piano': 25,
        'plant': 26,
        'radio': 27,
        'range_hood': 28,
        'sink': 29,
        'sofa': 30,
        'stairs': 31,
        'stool': 32,
        'table': 33,
        'tent': 34,
        'toilet': 35,
        'tv_stand': 36,
        'vase': 37,
        'wardrobe': 38,
        'xbox': 39
    }

    def __init__(self, num_points, partition='train', transformationType="Z", pertubation=False,
                 use_fps=True, class_choice=None):
        self.data, self.label = ModelNet.load_data(partition)
        if class_choice is not None:
            data = []
            label = []
            for class_name in class_choice:
                data_points, labels = self.get_with_shapeName(class_name)
                data.append(data_points)
                label.append(labels)
            self.data = np.concatenate(data, axis=0)
            self.label = np.concatenate(label, axis=0)
        print(self.data.shape)
        self.num_points = num_points
        self.partition = partition
        self.use_fps = use_fps
        self.use_perturbation = pertubation
        if transformationType == "Z":
            self.rot = PointCloudAzimuthalRotations()
        elif transformationType == "S":
            self.rot = PointCloudArbitraryRotation()
        else:
            self.rot = PointCloudBase()
        if self.use_perturbation:
            self.perturbation = PointCloudJitter()
        self.max_points = 2048

    def __getitem__(self, item):
        if self.use_fps is False:
            pt_idxes = np.arange(0, self.data[item].shape[0])
            np.random.shuffle(pt_idxes)
            point_cloud = self.data[item][pt_idxes[:self.num_points]].copy()
        else:
            point_cloud, fps_index = DataUtils.fps(self.data[item], ratio=0.5)
        label = self.label[item]
        point_cloud = self.rot(point_cloud)
        # if self.use_pertubation:
        #     pointcloud = self.pertubation(pointcloud)
        return point_cloud, label

    def get_reduced_points(self, data, num_pc):
        point_cloud, fps_index = DataUtils.fps(data, ratio=num_pc / self.max_points)
        return point_cloud

    def get_with_shapeName(self, name):
        result = np.where(self.label == self.shape_name[name])[0]
        labels = list(itertools.repeat(self.shape_name[name], len(result)))
        return self.data[result, :, :], labels

    @staticmethod
    def load_data(partition):
        ModelNet.download()
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        DATA_DIR = os.path.join(BASE_DIR, dataFile, 'ModelNet40')
        all_data = []
        all_label = []
        for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5' % partition)):
            f = h5py.File(h5_name, 'r')
            data = f['data'][:].astype('float32')
            label = f['label'][:].astype('int64')
            f.close()
            all_data.append(data)
            all_label.append(label)
        all_data = np.concatenate(all_data, axis=0)
        all_label = np.concatenate(all_label, axis=0)
        return all_data, all_label

    @staticmethod
    def download():
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        DATA_DIR = os.path.join(BASE_DIR, dataFile, 'ModelNet40')
        if not os.path.exists(DATA_DIR):
            os.makedirs(DATA_DIR, exist_ok=True)
        if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
            www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
            zipfile = os.path.basename(www)
            os.system('wget --no-check-certificate %s; unzip %s' % (www, zipfile))
            os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
            os.system('rm %s' % zipfile)


class IndoorSemantic(Dataset):
    def __int__(self):
        pass

    def __getitem__(self, item):
        pass

    @staticmethod
    def load_data(partition):
        pass

    @staticmethod
    def download():
        pass


class ShapeNet(Dataset):
    category_ids = {
        'Airplane': '02691156',
        'Bag': '02773838',
        'Cap': '02954340',
        'Car': '02958343',
        'Chair': '03001627',
        'Earphone': '03261776',
        'Guitar': '03467517',
        'Knife': '03624134',
        'Lamp': '03636649',
        'Laptop': '03642806',
        'Motorbike': '03790512',
        'Mug': '03797390',
        'Pistol': '03948459',
        'Rocket': '04099429',
        'Skateboard': '04225987',
        'Table': '04379243',
    }
    seg_classes = {
        'Airplane': [0, 1, 2, 3],
        'Bag': [4, 5],
        'Cap': [6, 7],
        'Car': [8, 9, 10, 11],
        'Chair': [12, 13, 14, 15],
        'Earphone': [16, 17, 18],
        'Guitar': [19, 20, 21],
        'Knife': [22, 23],
        'Lamp': [24, 25, 26, 27],
        'Laptop': [28, 29],
        'Motorbike': [30, 31, 32, 33, 34, 35],
        'Mug': [36, 37],
        'Pistol': [38, 39, 40],
        'Rocket': [41, 42, 43],
        'Skateboard': [44, 45, 46],
        'Table': [47, 48, 49],
    }

    def __init__(self, num_points=1024, partition='train', transforms=None, transformationType="Z", pertubation=False,
                 use_fps=True):
        self.data, self.label, self.pid = ShapeNet.load_data(partition)
        self.file = None
        self.use_fps = use_fps
        self.num_points = num_points
        self.y_mask = torch.zeros((len(self.seg_classes.keys()), 50), dtype=torch.bool)
        for i, labels in enumerate(self.seg_classes.values()):
            self.y_mask[i, labels] = 1

    def __getitem__(self, item):
        if self.use_fps is False:
            pt_idxes = np.arange(0, self.data[item].shape[0])
            np.random.shuffle(pt_idxes)
            point_cloud = self.data[item][pt_idxes[:self.num_points]].copy()
            pid = self.pid[item][pt_idxes[:self.num_points]].copy()
        else:
            ratio = int(self.data.shape[1] / self.num_points)
            point_cloud, fps_index = DataUtils.fps(self.data[item], ratio=ratio)
            pid = self.pid[item][fps_index]
        label = self.label[item]
        return point_cloud, label, pid

    @property
    def num_categories(self):
        return self.y_mask.shape[0]

    @property
    def num_classes(self):
        return self.y_mask.shape[1]

    def load_data(self, partition):
        ShapeNetDownload()
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        DATA_DIR = os.path.join(BASE_DIR, dataFile, 'ShapeNetPart')

        self.file = os.path.join(DATA_DIR, 'hdf5_data', '%s_hdf5_file_list.txt' % partition)
        all_data = []
        all_label = []
        all_part_id = []
        for h5_name in glob.glob(os.path.join(DATA_DIR, 'hdf5_data', 'ply_data_%s*.h5' % partition)):
            print(h5_name)
            f = h5py.File(h5_name, 'r')
            data = f['data'][:].astype('float32')
            label = f['label'][:].astype('int64')
            pid = f['pid'][:].astype('int64')
            f.close()
            all_data.append(data)
            all_label.append(label)
            all_part_id.append(pid)
        all_data = np.concatenate(all_data, axis=0)
        all_label = np.concatenate(all_label, axis=0)
        all_part_id = np.concatenate(all_part_id, axis=0)
        return all_data, all_label, all_part_id


if __name__ == '__main__':
    modelNet = ModelNet(num_points=1024, class_choice=["airplane", "car"])
    # loader = DataLoader(ModelNet(1024))
    # ModelNet.load_data("test")
    # print(dataFile)
    # point_completion = PointCompletionShapeNet(num_points=1024, partition="val")
    # point_completion.get_category()
    # point_completion.get_train_file()
    # point_completion.get_val_file()
