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


class FoldNet_Encoder(nn.Module):
    def __init__(self):
        super(FoldNet_Encoder, self).__init__()
        mlps1 = [3, 128, 256]
        self.first_mpl = nn.Sequential(*[nn.Sequential(torch.nn.Conv1d(mlps1[i - 1], mlps1[i], kernel_size=1),
                                                       torch.nn.BatchNorm1d(mlps1[i]),
                                                       torch.nn.ReLU()) for i in range(1, len(mlps1[:-1]))])
        self.first_mpl_last = nn.Sequential(torch.nn.Conv1d(mlps1[-2], mlps1[-1], kernel_size=1),
                                            torch.nn.BatchNorm1d(mlps1[-1]))
        ## Second  mlp
        mlps2 = [mlps1[-1] * 2, 512, 1024]
        self.second_mpl = nn.Sequential(*[nn.Sequential(torch.nn.Conv1d(mlps2[i - 1], mlps2[i], kernel_size=1),
                                                        torch.nn.BatchNorm1d(mlps2[i]),
                                                        torch.nn.ReLU()) for i in range(1, len(mlps2[:-1]))])
        self.second_mpl_last = nn.Sequential(torch.nn.Conv1d(mlps2[-2], mlps2[-1], kernel_size=1),
                                             torch.nn.BatchNorm1d(mlps2[-1]))

    def forward(self, xyz):
        xyz = xyz.permute(0, 2, 1)
        features = self.first_mpl(xyz)
        features = self.first_mpl_last(features)
        features_global = torch.max(features, dim=-1, keepdim=True)[0]  # 32*256*1
        features_global = features_global.repeat(1, 1, xyz.shape[2])  # 32*256*2048
        features = torch.cat([features, features_global], dim=1)
        features = self.second_mpl(features)
        features = self.second_mpl_last(features)
        features = torch.max(features, dim=-1)[0]  # 32*1024
        return features


class FoldNet_Decoder(nn.Module):
    def __init__(self):
        super(FoldNet_Decoder, self).__init__()

        self.grid_scale = 0.5
        self.num_points = 2048
        self.grid_size = int(np.sqrt(self.num_points))
        if self.grid_size ** 2 < self.num_points:
            self.grid_size += 1
        self.num_points = self.grid_size ** 2

        mlps1 = [1024 + 2, 512, 512]
        self.fold1 = nn.Sequential(*[nn.Sequential(torch.nn.Conv1d(mlps1[i - 1], mlps1[i], kernel_size=1),
                                                   torch.nn.BatchNorm1d(mlps1[i]),
                                                   torch.nn.ReLU()) for i in range(1, len(mlps1))])

        self.fold1_last_layer = torch.nn.Conv1d(512, 3, kernel_size=1)
        mlsp2 = [1024 + 3, 512, 512]
        self.fold2 = nn.Sequential(*[nn.Sequential(torch.nn.Conv1d(mlsp2[i - 1], mlsp2[i], kernel_size=1),
                                                   torch.nn.BatchNorm1d(mlsp2[i]),
                                                   torch.nn.ReLU()) for i in range(1, len(mlsp2))])
        self.fold2_last_layer = torch.nn.Conv1d(512, 3, kernel_size=1)

    def forward(self, features):
        grid_row = torch.linspace(-1 * self.grid_scale, self.grid_scale, self.grid_size).cuda()
        grid_col = torch.linspace(-1 * self.grid_scale, self.grid_scale, self.grid_size).cuda()
        grid = torch.meshgrid(grid_row, grid_col)
        grid = torch.reshape(torch.stack(grid, dim=2), (-1, 2)).unsqueeze(0)
        grid = grid.repeat(features.size(0), self.num_points // self.grid_size ** 2, 1)
        features = features.unsqueeze(dim=1).repeat(1, self.num_points, 1)
        new_features = torch.cat([features, grid], dim=2).transpose(2, 1)
        fold1 = self.fold1(new_features)
        fold1 = self.fold1_last_layer(fold1).transpose(2, 1)
        fold2_features = torch.cat([features, fold1], dim=2).transpose(2, 1)
        fold2 = self.fold2(fold2_features)
        fold2 = self.fold2_last_layer(fold2)
        return fold2.transpose(2, 1)


class FoldNetModel(BaseModel):
    def __init__(self, parameter, checkpoint_name, best_name, checkpoint_path, logger_file_name):
        super(FoldNetModel, self).__init__(parameter=parameter, checkpoint_name=checkpoint_name,
                                           best_name=best_name, checkpoint_path=checkpoint_path,
                                           logger_file_name=logger_file_name)
        self.encoder = FoldNet_Encoder()
        self.decoder = FoldNet_Decoder()
        self.l_cd = self.configure_loss_function()

    def forward(self, x):
        features = self.encoder(x)
        output = self.decoder(features)
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
        optimizer = optim.Adam([{'params': self.encoder.parameters()},
                                {'params': self.decoder.parameters()}],
                               lr=1e-3, betas=(0.9, 0.999))
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
            self.encoder = nn.DataParallel(self.encoder)
            self.decoder = nn.DataParallel(self.decoder)
            self.encoder.to(device)
            self.decoder.to(device)
        else:
            print("====> use only one cuda")
            self.encoder.to(device)
            self.decoder.to(device)

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
            "model_encoder_state": self.encoder.state_dict(),
            "model_decoder_state": self.decoder.state_dict(),
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
                    "model_encoder_state": self.encoder.state_dict(),
                    "model_decoder_state": self.decoder.state_dict(),
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
