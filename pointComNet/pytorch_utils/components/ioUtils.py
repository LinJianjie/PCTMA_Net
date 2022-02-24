import os
import sys
import open3d as o3d

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
import torch
import shutil


class IOStream:
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text + '\n')
        self.f.flush()

    def close(self):
        self.f.close()


def readText(path, sep):
    f = open(path, 'r')
    data = []
    for line in f.readlines():
        ll = line.replace('\n', '').split(sep)
        data.append(ll)
    return data


def save_ply(pc, path):
    """
        Save numpy tensor as .ply file
        :param pc -> numpy ndarry with shape [num_points, 3] expected
        :param path -> saving path including .ply name
        """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    o3d.io.write_point_cloud(path, pcd)


def save_batch_ply(pc, path):
    for i in range(pc.shape[0]):
        save_ply(pc[0], path)


def save_checkpoint(state, best_model_name):
    save_path = os.path.join(os.path.dirname(__file__), "../../checkpoints", best_model_name)
    torch.save(state, save_path)


def load_checkpoint(model=None, optimizer=None, filename="checkpoint"):
    # filename = "{}.pth.tar".format(filename)

    if os.path.isfile(filename):
        print("==> Loading from checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        epoch = checkpoint["epoch"]
        if model is not None and checkpoint["model_state"] is not None:
            model.load_state_dict(checkpoint["model_state"])
        if optimizer is not None and checkpoint["optimizer_state"] is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        print("==> Done")
        return None
    else:
        print("==> Checkpoint '{}' not found".format(filename))
        return None


def make_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def make_dirs_checkout():
    checkout_path = os.path.join(os.getcwd(), "checkpoints")
    make_dirs(path=checkout_path)
    return checkout_path


def make_dirs_log():
    log_dirs = os.path.join(os.getcwd(), "logs")
    make_dirs(path=log_dirs)
    return log_dirs


def copy_file(source, des, filename, rename=None):
    shutil.copy(source, des)
    old_name = os.path.join(des, filename)
    new_name = os.path.join(des, rename)
    os.rename(old_name, new_name)
