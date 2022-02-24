import sys
from pathlib import Path

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from pointComNet.utils.loss_function import EMDLoss, CDLoss
import numpy as np
from pointComNet.pytorch_utils.components.pytorch_base_model import BaseModel
import torch.optim as optim
import itertools
from collections import OrderedDict
from pointComNet.pytorch_utils.components.Logger import *
from pointComNet.pytorch_utils.components.ioUtils import save_ply, make_dirs, copy_file
from pointComNet.pytorch_utils.components.dataSet import PointCompletionShapeNet


def knn(x, k):
    batch_size = x.size(0)
    num_points = x.size(2)

    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)

    if idx.get_device() == -1:
        idx_base = torch.arange(0, batch_size).view(-1, 1, 1) * num_points
    else:
        idx_base = torch.arange(0, batch_size, device=idx.get_device()).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)

    return idx


def local_cov(pts, idx):
    batch_size = pts.size(0)
    num_points = pts.size(2)
    pts = pts.view(batch_size, -1, num_points)  # (batch_size, 3, num_points)

    _, num_dims, _ = pts.size()

    x = pts.transpose(2, 1).contiguous()  # (batch_size, num_points, 3)
    x = x.view(batch_size * num_points, -1)[idx, :]  # (batch_size*num_points*2, 3)
    x = x.view(batch_size, num_points, -1, num_dims)  # (batch_size, num_points, k, 3)

    x = torch.matmul(x[:, :, 0].unsqueeze(3), x[:, :, 1].unsqueeze(
        2))  # (batch_size, num_points, 3, 1) * (batch_size, num_points, 1, 3) -> (batch_size, num_points, 3, 3)
    # x = torch.matmul(x[:,:,1:].transpose(3, 2), x[:,:,1:])
    x = x.view(batch_size, num_points, 9).transpose(2, 1)  # (batch_size, 9, num_points)

    x = torch.cat((pts, x), dim=1)  # (batch_size, 12, num_points)

    return x


def local_maxpool(x, idx):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()  # (batch_size, num_points, num_dims)
    x = x.view(batch_size * num_points, -1)[idx, :]  # (batch_size*n, num_dims) -> (batch_size*n*k, num_dims)
    x = x.view(batch_size, num_points, -1, num_dims)  # (batch_size, num_points, k, num_dims)
    x, _ = torch.max(x, dim=2)  # (batch_size, num_points, num_dims)

    return x


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)  # (batch_size, num_dims, num_points)
    if idx is None:
        idx = knn(x, k=k)  # (batch_size, num_points, k)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()  # (batch_size, num_points, num_dims)
    feature = x.view(batch_size * num_points, -1)[idx, :]  # (batch_size*n, num_dims) -> (batch_size*n*k, num_dims)
    feature = feature.view(batch_size, num_points, k, num_dims)  # (batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)  # (batch_size, num_points, k, num_dims)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1,
                                                         2)  # (batch_size, num_points, k, 2*num_dims) -> (batch_size, 2*num_dims, num_points, k)

    return feature  # (batch_size, 2*num_dims, num_points, k)


class FoldNet_Encoder(nn.Module):
    def __init__(self, args):
        super(FoldNet_Encoder, self).__init__()
        self.k = args["k"]
        self.n = 2048  # input point cloud size
        self.mlp1 = nn.Sequential(
            nn.Conv1d(12, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.ReLU(),
        )
        self.linear1 = nn.Linear(64, 64)
        self.conv1 = nn.Conv1d(64, 128, 1)
        self.linear2 = nn.Linear(128, 128)
        self.conv2 = nn.Conv1d(128, 1024, 1)
        self.mlp2 = nn.Sequential(
            nn.Conv1d(1024, args["feat_dims"], 1),
            nn.ReLU(),
            nn.Conv1d(args["feat_dims"], args["feat_dims"], 1),
        )

    def graph_layer(self, x, idx):
        x = local_maxpool(x, idx)
        x = self.linear1(x)
        x = x.transpose(2, 1)
        x = F.relu(self.conv1(x))
        x = local_maxpool(x, idx)
        x = self.linear2(x)
        x = x.transpose(2, 1)
        x = self.conv2(x)
        return x

    def forward(self, pts):
        pts = pts.transpose(2, 1)  # (batch_size, 3, num_points)
        idx = knn(pts, k=self.k)
        x = local_cov(pts, idx)  # (batch_size, 3, num_points) -> (batch_size, 12, num_points])
        x = self.mlp1(x)  # (batch_size, 12, num_points) -> (batch_size, 64, num_points])
        x = self.graph_layer(x, idx)  # (batch_size, 64, num_points) -> (batch_size, 1024, num_points)
        x = torch.max(x, 2, keepdim=True)[0]  # (batch_size, 1024, num_points) -> (batch_size, 1024, 1)
        x = self.mlp2(x)  # (batch_size, 1024, 1) -> (batch_size, feat_dims, 1)
        feat = x.transpose(2, 1)  # (batch_size, feat_dims, 1) -> (batch_size, 1, feat_dims)
        return feat


class FoldNet_Decoder(nn.Module):
    def __init__(self, args):
        super(FoldNet_Decoder, self).__init__()
        self.m = 2025  # 45 * 45.
        self.shape = args["shape"]
        self.meshgrid = [[-0.3, 0.3, 45], [-0.3, 0.3, 45]]
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        self.sphere = np.load(os.path.join(BASE_DIR, "sphere.npy"))
        self.gaussian = np.load(os.path.join(BASE_DIR, "gaussian.npy"))
        if self.shape == 'plane':
            self.folding1 = nn.Sequential(
                nn.Conv1d(args["feat_dims"] + 2, args["feat_dims"], 1),
                nn.ReLU(),
                nn.Conv1d(args["feat_dims"], args["feat_dims"], 1),
                nn.ReLU(),
                nn.Conv1d(args["feat_dims"], 3, 1),
            )
        else:
            self.folding1 = nn.Sequential(
                nn.Conv1d(args["feat_dims"] + 3, args["feat_dims"], 1),
                nn.ReLU(),
                nn.Conv1d(args["feat_dims"], args["feat_dims"], 1),
                nn.ReLU(),
                nn.Conv1d(args["feat_dims"], 3, 1),
            )
        self.folding2 = nn.Sequential(
            nn.Conv1d(args["feat_dims"] + 3, args["feat_dims"], 1),
            nn.ReLU(),
            nn.Conv1d(args["feat_dims"], args["feat_dims"], 1),
            nn.ReLU(),
            nn.Conv1d(args["feat_dims"], 3, 1),
        )

    def build_grid(self, batch_size):
        if self.shape == 'plane':
            x = np.linspace(*self.meshgrid[0])
            y = np.linspace(*self.meshgrid[1])
            points = np.array(list(itertools.product(x, y)))
        elif self.shape == 'sphere':
            points = self.sphere
        elif self.shape == 'gaussian':
            points = self.gaussian
        points = np.repeat(points[np.newaxis, ...], repeats=batch_size, axis=0)
        points = torch.tensor(points)
        return points.float()

    def forward(self, x):
        x = x.transpose(1, 2).repeat(1, 1, self.m)  # (batch_size, feat_dims, num_points)
        points = self.build_grid(x.shape[0]).transpose(1,
                                                       2)  # (batch_size, 2, num_points) or (batch_size, 3, num_points)
        if x.get_device() != -1:
            points = points.cuda(x.get_device())
        cat1 = torch.cat((x, points),
                         dim=1)  # (batch_size, feat_dims+2, num_points) or (batch_size, feat_dims+3, num_points)
        folding_result1 = self.folding1(cat1)  # (batch_size, 3, num_points)
        cat2 = torch.cat((x, folding_result1), dim=1)  # (batch_size, 515, num_points)
        folding_result2 = self.folding2(cat2)  # (batch_size, 3, num_points)
        return folding_result2.transpose(1, 2)  # (batch_size, num_points ,3)


class FoldNet(nn.Module):
    def __init__(self, parameter):
        super(FoldNet, self).__init__()
        self.encoder = FoldNet_Encoder(args=parameter)
        self.decoder = FoldNet_Decoder(args=parameter)

    def forward(self, x):
        feature = self.encoder(x)
        output = self.decoder(feature)
        return output


class FoldNetModel(BaseModel):
    def __init__(self, parameter, checkpoint_name, best_name, checkpoint_path, logger_file_name):
        super(FoldNetModel, self).__init__(parameter=parameter, checkpoint_name=checkpoint_name,
                                           best_name=best_name, checkpoint_path=checkpoint_path,
                                           logger_file_name=logger_file_name)
        self.model = FoldNet(parameter=parameter)
        self.l_cd = self.configure_loss_function()

    def forward(self, x):
        output = self.model(x)
        return output

    def backward_model(self, output, gt):
        loss = self.l_cd(output, gt)
        return loss, loss

    def backward_model_update(self, loss):
        self.model_optimizer["opt"].zero_grad()
        loss.backward()
        self.model_optimizer["opt"].step()
        self.model_optimizer["scheduler"].step()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3, betas=(0.9, 0.999))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.parameter["epochs"],
                                                               eta_min=1e-3)
        opt_scheduler = OrderedDict({"opt": optimizer, "scheduler": scheduler})
        self.model_optimizer = opt_scheduler
        return opt_scheduler

    @staticmethod
    def configure_loss_function():
        pointCloud_reconstruction_loss = CDLoss()
        return pointCloud_reconstruction_loss

    def toCuda(self, device):
        self.device = device
        if torch.cuda.device_count() > 1:
            print("====> use data parallel")
            self.model = nn.DataParallel(self.model)
            self.model.to(device)
        else:
            print("====> use only one cuda")
            self.model.to(device)

    def load_checkpoint(self, filename="checkpoint"):
        pass

    def train_one_epoch(self, train_loader, at_epoch, n_epochs, batch_size):
        count_ae = 0
        train_loss_ae = 0
        train_loss_cd = 0
        self.train()
        for i, dataset in enumerate(train_loader):
            gt_point_cloud, partial_point_cloud, label_point_cloud = dataset
            gt_point_cloud = gt_point_cloud.to(self.device)
            partial_point_cloud = partial_point_cloud.to(self.device)
            out_px1 = self(partial_point_cloud)
            gt1 = gt_point_cloud
            out_px1_generated = out_px1
            loss_px_gt, loss_cd = self.backward_model(output=out_px1_generated, gt=gt1)
            loss_pct = loss_px_gt
            self.backward_model_update(loss=loss_pct)
            count_ae += 1
            train_loss_ae += loss_pct.item()
            train_loss_cd += loss_cd.item()
        return train_loss_ae / count_ae, train_loss_cd / count_ae

    def train_step(self, start_epoch, n_epochs, train_loader, test_loader, best_loss=0.0, batch_size=8,
                   best_model_name="best_model.pth"):
        loss_ae = 1000
        loss_cd = 1000
        state = {
            "batch_size": batch_size,
            "model_ae_loss": loss_ae,
            "model_state": self.model.state_dict(),
            "optimizer_pct_state": self.model_optimizer["opt"].state_dict()
        }
        cd_sparse = 1000
        save_results = False
        for at_epoch in range(start_epoch, n_epochs):
            loss_ae_updated, loss_cd_update = self.train_one_epoch(train_loader=train_loader,
                                                                   at_epoch=at_epoch,
                                                                   n_epochs=n_epochs,
                                                                   batch_size=batch_size)
            cd_sparse_update = self.evaluation_step(test_loader=test_loader)
            if cd_sparse_update < cd_sparse:
                cd_sparse = cd_sparse_update
                save_results = True
            self.Logger.INFO('Train %d/%d, loss_ae: %.6f, loss_cd: %.6f, best cd: %.6f\n', at_epoch, n_epochs,
                             loss_ae_updated, loss_cd_update, cd_sparse)
            if loss_cd_update < loss_cd and loss_cd_update < 0.004 and save_results:
                loss_cd = loss_cd_update
                save_results = False
                state = {
                    "batch_size": batch_size,
                    "model_ae_loss": loss_cd,
                    "model_state": self.model.state_dict(),
                    "optimizer_pct_state": self.model_optimizer["opt"].state_dict()
                }
                self.save_checkpoint(state=state, best_model_name=best_model_name)

    def evaluation_step(self, test_loader, check_point_name=None):
        evaluate_loss_sparse = []
        evaluate_loss_dense = []
        evaluate_class_choice_sparse = {"Plane": [], "Cabinet": [], "Car": [], "Chair": [], "Lamp": [], "Couch": [],
                                        "Table": [], "Watercraft": []}
        evaluate_class_choice_dense = {"Plane": [], "Cabinet": [], "Car": [], "Chair": [], "Lamp": [], "Couch": [],
                                       "Table": [], "Watercraft": []}
        count = 0.0
        if self.parameter["gene_file"]:
            save_ply_path = os.path.join(os.path.dirname(__file__), "../../save_ae_ply_data")
            make_dirs(save_ply_path)
            if check_point_name is not None:
                check_point_base_name = check_point_name.split(".")
                save_ply_path = os.path.join(os.path.dirname(__file__), "../../save_ae_ply_data",
                                             check_point_base_name[0])
                make_dirs(save_ply_path)
            count_k = 0
        for i, dataset in enumerate(test_loader):
            gt_point_cloud, partial_point_cloud, label_point_cloud = dataset
            gt_point_cloud = gt_point_cloud.to(self.device)
            partial_point_cloud = partial_point_cloud.to(self.device)
            with torch.no_grad():
                self.eval()
                out_px1 = self(partial_point_cloud)
                cd = self.l_cd(out_px1, gt_point_cloud)
                evaluate_loss_sparse.append(cd.item())
                for k in range(partial_point_cloud.shape[0]):
                    class_name_choice = PointCompletionShapeNet.evaluation_class(
                        label_name=test_loader.dataset.label_to_category(label_point_cloud[k]))
                    evaluate_class_choice_sparse[class_name_choice].append(cd.item())

                if self.parameter["gene_file"]:
                    for k in range(partial_point_cloud.shape[0]):
                        base_name = test_loader.dataset.label_to_category(label_point_cloud[k]) + "_" + str(
                            count_k) + "_pc_recon" + ".ply"
                        template_path = os.path.join(save_ply_path, base_name)
                        save_ply(pc=out_px1[k].cpu().detach().numpy(), path=template_path)

                        base_name = test_loader.dataset.label_to_category(label_point_cloud[k]) + "_" + str(
                            count_k) + "_pc" + ".ply"
                        template_path = os.path.join(save_ply_path, base_name)
                        save_ply(partial_point_cloud[k].cpu().detach().numpy(), path=template_path)

                        base_name = test_loader.dataset.label_to_category(label_point_cloud[k]) + "_" + str(
                            count_k) + "_gt" + ".ply"
                        template_path = os.path.join(save_ply_path, base_name)
                        save_ply(gt_point_cloud[k].cpu().detach().numpy(), path=template_path)

                        count_k += 1
        for key, item in evaluate_class_choice_sparse.items():
            if item:
                evaluate_class_choice_sparse[key] = sum(item) / len(item)

        self.Logger.INFO(
            '====> cd_sparse: Airplane: %.6f, Cabinet: %.6f, Car: %.6f, Chair: %.6f, Lamp: %.6f, Sofa: %.6f, Table: %.6f, Watercraft: %.6f, mean: %.6f',
            evaluate_class_choice_sparse["Plane"], evaluate_class_choice_sparse["Cabinet"],
            evaluate_class_choice_sparse["Car"], evaluate_class_choice_sparse["Chair"],
            evaluate_class_choice_sparse["Lamp"], evaluate_class_choice_sparse["Couch"],
            evaluate_class_choice_sparse["Table"], evaluate_class_choice_sparse["Watercraft"],
            sum(evaluate_loss_sparse) / len(evaluate_loss_sparse))

        return sum(evaluate_loss_sparse) / len(evaluate_loss_sparse)
