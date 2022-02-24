# -*- coding: utf-8 -*-
# @Author: Haozhe Xie
# @Date:   2019-09-06 11:35:30
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2020-02-22 19:20:36
# @Email:  cshzxie@gmail.com

import torch
import torch.nn as nn
import sys
from pathlib import Path
import os

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from pointComNet.utils.loss_function import Gridding_function, Gridding_Reverse_function, CubicFeatureSampling_function, \
    CDLoss
from pointComNet.pytorch_utils.components.pytorch_base_model import BaseModel
import torch.optim as optim
from collections import OrderedDict
from pointComNet.pytorch_utils.components.dataSet import PointCompletionShapeNet
from pointComNet.pytorch_utils.components.torch_cluster_sampling import farthest_point_sampling
from pointComNet.pytorch_utils.components.ioUtils import save_ply, make_dirs, copy_file


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


class GRNet(torch.nn.Module):
    def __init__(self):
        super(GRNet, self).__init__()
        self.gridding = Gridding_function(scale=64)
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv3d(1, 32, kernel_size=4, padding=2),
            torch.nn.BatchNorm3d(32),
            torch.nn.LeakyReLU(0.2),
            torch.nn.MaxPool3d(kernel_size=2)
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv3d(32, 64, kernel_size=4, padding=2),
            torch.nn.BatchNorm3d(64),
            torch.nn.LeakyReLU(0.2),
            torch.nn.MaxPool3d(kernel_size=2)
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv3d(64, 128, kernel_size=4, padding=2),
            torch.nn.BatchNorm3d(128),
            torch.nn.LeakyReLU(0.2),
            torch.nn.MaxPool3d(kernel_size=2)
        )
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv3d(128, 256, kernel_size=4, padding=2),
            torch.nn.BatchNorm3d(256),
            torch.nn.LeakyReLU(0.2),
            torch.nn.MaxPool3d(kernel_size=2)
        )
        self.fc5 = torch.nn.Sequential(
            torch.nn.Linear(16384, 2048),
            torch.nn.ReLU()
        )
        self.fc6 = torch.nn.Sequential(
            torch.nn.Linear(2048, 16384),
            torch.nn.ReLU()
        )
        self.dconv7 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, bias=False, padding=1),
            torch.nn.BatchNorm3d(128),
            torch.nn.ReLU()
        )
        self.dconv8 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, bias=False, padding=1),
            torch.nn.BatchNorm3d(64),
            torch.nn.ReLU()
        )
        self.dconv9 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, bias=False, padding=1),
            torch.nn.BatchNorm3d(32),
            torch.nn.ReLU()
        )
        self.dconv10 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(32, 1, kernel_size=4, stride=2, bias=False, padding=1),
            torch.nn.BatchNorm3d(1),
            torch.nn.ReLU()
        )
        self.gridding_rev = Gridding_Reverse_function(scale=64)
        self.point_sampling = RandomPointSampling(n_points=2048)
        self.feature_sampling = CubicFeatureSampling_function()
        self.fc11 = torch.nn.Sequential(
            torch.nn.Linear(1792, 1792),
            torch.nn.ReLU()
        )
        self.fc12 = torch.nn.Sequential(
            torch.nn.Linear(1792, 448),
            torch.nn.ReLU()
        )
        self.fc13 = torch.nn.Sequential(
            torch.nn.Linear(448, 112),
            torch.nn.ReLU()
        )
        self.fc14 = torch.nn.Linear(112, 24)

    def forward(self, data):
        partial_cloud = data
        # print(partial_cloud.size())     # torch.Size([batch_size, 2048, 3])
        pt_features_64_l = self.gridding(partial_cloud).view(-1, 1, 64, 64, 64)
        # print(pt_features_64_l.size())  # torch.Size([batch_size, 1, 64, 64, 64])
        pt_features_32_l = self.conv1(pt_features_64_l)
        # print(pt_features_32_l.size())  # torch.Size([batch_size, 32, 32, 32, 32])
        pt_features_16_l = self.conv2(pt_features_32_l)
        # print(pt_features_16_l.size())  # torch.Size([batch_size, 64, 16, 16, 16])
        pt_features_8_l = self.conv3(pt_features_16_l)
        # print(pt_features_8_l.size())   # torch.Size([batch_size, 128, 8, 8, 8])
        pt_features_4_l = self.conv4(pt_features_8_l)
        # print(pt_features_4_l.size())   # torch.Size([batch_size, 256, 4, 4, 4])
        features = self.fc5(pt_features_4_l.view(-1, 16384))
        # print(features.size())          # torch.Size([batch_size, 2048])
        pt_features_4_r = self.fc6(features).view(-1, 256, 4, 4, 4) + pt_features_4_l
        # print(pt_features_4_r.size())   # torch.Size([batch_size, 256, 4, 4, 4])
        pt_features_8_r = self.dconv7(pt_features_4_r) + pt_features_8_l
        # print(pt_features_8_r.size())   # torch.Size([batch_size, 128, 8, 8, 8])
        pt_features_16_r = self.dconv8(pt_features_8_r) + pt_features_16_l
        # print(pt_features_16_r.size())  # torch.Size([batch_size, 64, 16, 16, 16])
        pt_features_32_r = self.dconv9(pt_features_16_r) + pt_features_32_l
        # print(pt_features_32_r.size())  # torch.Size([batch_size, 32, 32, 32, 32])
        pt_features_64_r = self.dconv10(pt_features_32_r) + pt_features_64_l
        # print(pt_features_64_r.size())  # torch.Size([batch_size, 1, 64, 64, 64])
        sparse_cloud = self.gridding_rev(pt_features_64_r.squeeze(dim=1))
        # print(sparse_cloud.size())      # torch.Size([batch_size, 262144, 3])
        sparse_cloud = self.point_sampling(sparse_cloud, partial_cloud)
        # print(sparse_cloud.size())      # torch.Size([batch_size, 2048, 3])
        point_features_32 = self.feature_sampling(sparse_cloud, pt_features_32_r).view(-1, 2048, 256)
        # print(point_features_32.size()) # torch.Size([batch_size, 2048, 256])
        point_features_16 = self.feature_sampling(sparse_cloud, pt_features_16_r).view(-1, 2048, 512)
        # print(point_features_16.size()) # torch.Size([batch_size, 2048, 512])
        point_features_8 = self.feature_sampling(sparse_cloud, pt_features_8_r).view(-1, 2048, 1024)
        # print(point_features_8.size())  # torch.Size([batch_size, 2048, 1024])
        point_features = torch.cat([point_features_32, point_features_16, point_features_8], dim=2)
        # print(point_features.size())    # torch.Size([batch_size, 2048, 1792])
        point_features = self.fc11(point_features)
        # print(point_features.size())    # torch.Size([batch_size, 2048, 1792])
        point_features = self.fc12(point_features)
        # print(point_features.size())    # torch.Size([batch_size, 2048, 448])
        point_features = self.fc13(point_features)
        # print(point_features.size())    # torch.Size([batch_size, 2048, 112])
        point_offset = self.fc14(point_features).view(-1, 16384, 3)
        # print(point_features.size())    # torch.Size([batch_size, 16384, 3])
        dense_cloud = sparse_cloud.unsqueeze(dim=2).repeat(1, 1, 8, 1).view(-1, 16384, 3) + point_offset
        # print(dense_cloud.size())       # torch.Size([batch_size, 16384, 3])

        return sparse_cloud, dense_cloud


class GRNetModel(BaseModel):
    def __init__(self, parameter, checkpoint_name, best_name, checkpoint_path, logger_file_name):
        super(GRNetModel, self).__init__(parameter=parameter, checkpoint_name=checkpoint_name, best_name=best_name,
                                         checkpoint_path=checkpoint_path, logger_file_name=logger_file_name)
        self.model = GRNet()
        self.l_cd = self.configure_loss_function()

    def forward(self, x):
        sparse_cloud, dense_cloud = self.model(x)
        return sparse_cloud, dense_cloud

    def configure_optimizers(self):
        optimizer_pct = optim.Adam(self.parameters(), lr=1e-3, betas=(0.9, 0.999))
        scheduler_pct = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_pct, self.parameter["epochs"],
                                                                   eta_min=1e-3)
        # scheduler_pct = torch.optim.lr_scheduler.StepLR(optimizer=optimizer_pct, step_size=40, gamma=0.1)
        opt_pct = OrderedDict({"opt": optimizer_pct, "scheduler": scheduler_pct})
        self.model_optimizer = opt_pct
        return opt_pct

    @staticmethod
    def configure_loss_function():
        pointCloud_reconstruction_loss = CDLoss()
        return pointCloud_reconstruction_loss

    def backward_model(self, sparse_cloud, dense_cloud, gt):
        cd_sparse = self.l_cd(sparse_cloud, gt)
        cd_dense = self.l_cd(dense_cloud, gt)
        return cd_dense + cd_sparse, cd_dense

    #
    def backward_model_update(self, loss):
        self.model_optimizer["opt"].zero_grad()
        loss.backward()
        self.model_optimizer["opt"].step()
        self.model_optimizer["scheduler"].step()

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
        # filename = "{}.pth.tar".format(filename)
        if os.path.isfile(filename):
            print("==> Loading from checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename)
            if checkpoint["model_ae_loss"] is not None:
                print("model_ae_loss: ", checkpoint["model_ae_loss"])
            if checkpoint["model_grNet_state"] is not None:
                self.model.load_state_dict(checkpoint["model_grNet_state"])
            if checkpoint["optimizer_state"] is not None:
                self.model_optimizer["opt"].load_state_dict(checkpoint["optimizer_state"])
            print("==> Done")
            return None
        else:
            print("==> Checkpoint '{}' not found".format(filename))
            return None

    def train_one_epoch(self, train_loader, at_epoch, n_epochs, batch_size):
        count_ae = 0
        train_loss_ae = 0
        train_loss_cd = 0
        self.train()
        for i, dataset in enumerate(train_loader):
            gt_point_cloud, partial_point_cloud, label_point_cloud = dataset
            gt_point_cloud = gt_point_cloud.to(self.device)
            partial_point_cloud = partial_point_cloud.to(self.device)
            # if self.train_gt:
            #     out_gt_3, out_gt_2, out_gt_1 = self(gt_point_cloud)
            #     loss_gt, loss_cd = self.backward_pct()
            #     loss_pct = loss_gt
            # if self.train_pt:
            #     out_px = self(partial_point_cloud)
            #     loss_px, loss_cd = self.backward_pct(output_decode=out_px, target=partial_point_cloud)
            #     loss_pct = loss_px
            # if self.train_pt_to_gt:
            sparse_cloud, dense_cloud = self(partial_point_cloud)
            gt1 = gt_point_cloud
            loss_grnet, loss_cd_dense = self.backward_model(sparse_cloud=sparse_cloud, dense_cloud=dense_cloud,
                                                            gt=gt_point_cloud)
            # if self.train_gt and self.train_pt:
            #     loss_pct = (loss_gt + loss_px) / 2.0
            # loss_pct = (loss_px + loss_gt) / 2.0
            self.backward_model_update(loss=loss_grnet)
            count_ae += 1
            train_loss_ae += loss_grnet.item()
            train_loss_cd += loss_cd_dense.item()
        # self.Logger.INFO('Train %d/%d, loss_ae: %.6f, loss_cd: %.6f', at_epoch, n_epochs, train_loss_ae / count_ae,
        #                  train_loss_cd / count_ae)
        return train_loss_ae / count_ae, train_loss_cd / count_ae

    def train_step(self, start_epoch, n_epochs, train_loader, test_loader, best_loss=0.0, batch_size=8,
                   best_model_name="best_model.pth"):
        self.count_parameters()
        loss_ae = 1000
        loss_cd = 1000
        state = {
            "batch_size": batch_size,
            "model_ae_loss": loss_cd,
            "model_grNet_state": self.model.state_dict(),
            "optimizer_state": self.model_optimizer["opt"].state_dict()
        }
        cd_sparse = 1000
        cd_dense = 1000
        save_file = False
        for at_epoch in range(start_epoch, n_epochs):
            loss_ae_updated, loss_cd_update = self.train_one_epoch(train_loader=train_loader,
                                                                   at_epoch=at_epoch,
                                                                   n_epochs=n_epochs,
                                                                   batch_size=batch_size)
            cd_sparse_update, cd_dense_update = self.evaluation_step(test_loader=test_loader)
            if cd_dense_update < cd_dense:
                cd_sparse = cd_sparse_update
                cd_dense = cd_dense_update
                save_file = True

            self.Logger.INFO('Train %d/%d, loss_ae: %.6f, loss_cd: %.4f, best cd_sparse: %.4f, best cd_dense: %.4f\n',
                             at_epoch, n_epochs, loss_ae_updated, loss_cd_update, cd_sparse * 10000, cd_dense * 10000)

            if save_file and cd_dense < 0.002:
                loss_cd = loss_cd_update
                save_file = False
                state = {
                    "batch_size": batch_size,
                    "model_ae_loss": loss_cd,
                    "model_grNet_state": self.model.state_dict(),
                    "optimizer_state": self.model_optimizer["opt"].state_dict()
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
                sparse_cloud, dense_cloud = self(partial_point_cloud)
                sparse_cloud, _, _ = farthest_point_sampling(dense_cloud, ratio=0.125)
                cd_sparse = self.l_cd(sparse_cloud, gt_point_cloud)
                evaluate_loss_sparse.append(cd_sparse.item())
                cd_dense = self.l_cd(dense_cloud, gt_point_cloud)
                evaluate_loss_dense.append(cd_dense.item())
                for k in range(partial_point_cloud.shape[0]):
                    class_name_choice = PointCompletionShapeNet.evaluation_class(
                        label_name=test_loader.dataset.label_to_category(label_point_cloud[k]))
                    evaluate_class_choice_sparse[class_name_choice].append(cd_sparse.item())
                    evaluate_class_choice_dense[class_name_choice].append(cd_dense.item())

                if self.parameter["gene_file"]:
                    for k in range(partial_point_cloud.shape[0]):
                        base_name = test_loader.dataset.label_to_category(label_point_cloud[k]) + "_" + str(
                            count_k) + "_pc_recon" + ".ply"
                        template_path = os.path.join(save_ply_path, base_name)
                        save_ply(pc=sparse_cloud[k].cpu().detach().numpy(), path=template_path)

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

        for key, item in evaluate_class_choice_dense.items():
            if item:
                evaluate_class_choice_dense[key] = sum(item) / len(item)

        self.Logger.INFO(
            '====> cd_sparse: Airplane: %.4f, Cabinet: %.4f, Car: %.4f, Chair: %.4f, Lamp: %.4f, Sofa: %.4f, Table: %.4f, Watercraft: %.4f, mean: %.4f',
            evaluate_class_choice_sparse["Plane"] * 10000, evaluate_class_choice_sparse["Cabinet"] * 10000,
            evaluate_class_choice_sparse["Car"] * 10000, evaluate_class_choice_sparse["Chair"] * 10000,
            evaluate_class_choice_sparse["Lamp"] * 10000, evaluate_class_choice_sparse["Couch"] * 10000,
            evaluate_class_choice_sparse["Table"] * 10000, evaluate_class_choice_sparse["Watercraft"] * 10000,
            sum(evaluate_loss_sparse) / len(evaluate_loss_sparse) * 10000)

        self.Logger.INFO(
            '====> cd_dense: Airplane: %.4f, Cabinet: %.4f, Car: %.4f, Chair: %.4f, Lamp: %.4f, Sofa: %.4f, Table: %.4f, Watercraft: %.4f, mean: %.4f',
            evaluate_class_choice_dense["Plane"] * 10000, evaluate_class_choice_dense["Cabinet"] * 10000,
            evaluate_class_choice_dense["Car"] * 10000, evaluate_class_choice_dense["Chair"] * 10000,
            evaluate_class_choice_dense["Lamp"] * 10000, evaluate_class_choice_dense["Couch"] * 10000,
            evaluate_class_choice_dense["Table"] * 10000, evaluate_class_choice_dense["Watercraft"] * 10000,
            sum(evaluate_loss_dense) / len(evaluate_loss_dense) * 10000)

        return sum(evaluate_loss_sparse) / len(evaluate_loss_sparse), sum(evaluate_loss_dense) / len(
            evaluate_loss_dense)

    def evaluation_step_kitti(self, test_loader, check_point_name=None):
        if True:
            save_ply_path = os.path.join(os.path.dirname(__file__), "../../save_kitti_ply_data")
            make_dirs(save_ply_path)
            if check_point_name is not None:
                check_point_base_name = check_point_name.split(".")
                save_ply_path = os.path.join(os.path.dirname(__file__), "../../save_kitti_ply_data",
                                             check_point_base_name[0])
                make_dirs(save_ply_path)
        count_k = 0

        for i, dataset in enumerate(test_loader):
            gt_point_cloud, partial_point_cloud, label_point_cloud = dataset
            gt_point_cloud = gt_point_cloud.to(self.device)
            partial_point_cloud = partial_point_cloud.to(self.device)
            if count_k >= 100:
                return
            with torch.no_grad():
                self.eval()
                sparse_cloud, dense_cloud = self(partial_point_cloud)
                pc_point_completion_dense = dense_cloud
                if True:
                    for k in range(partial_point_cloud.shape[0]):
                        base_name = "kitti_car" + "_" + str(count_k) + "_pc_recon" + ".ply"
                        template_path = os.path.join(save_ply_path, base_name)
                        save_ply(pc=pc_point_completion_dense[k].cpu().detach().numpy(), path=template_path)

                        base_name = "kitti_car" + "_" + str(count_k) + "_pc" + ".ply"
                        template_path = os.path.join(save_ply_path, base_name)
                        save_ply(partial_point_cloud[k].cpu().detach().numpy(), path=template_path)
                        count_k += 1
