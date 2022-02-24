import sys
from pathlib import Path

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import torch
from torch.autograd import Variable
import torch.nn as nn
from common import PointNetfeat
from pointComNet.utils.loss_function import EMDLoss, CDLoss
import numpy as np
from pointComNet.pytorch_utils.components.pytorch_base_model import BaseModel
import torch.optim as optim
from collections import OrderedDict
from pointComNet.pytorch_utils.components.Logger import *
from pointComNet.pytorch_utils.components.ioUtils import save_ply, make_dirs, copy_file
from pointComNet.pytorch_utils.components.dataSet import PointCompletionShapeNet


class PointNetFACEModel(BaseModel):
    def __init__(self, parameter, checkpoint_name, best_name, checkpoint_path, logger_file_name):
        super(PointNetFACEModel, self).__init__(parameter=parameter, checkpoint_name=checkpoint_name,
                                                best_name=best_name,
                                                checkpoint_path=checkpoint_path, logger_file_name=logger_file_name)
        self.num_points = self.parameter["num_points"]
        print("the number of points: ", self.num_points)
        self.output_channels = self.parameter["output_channels"]
        self.encoder = nn.Sequential(
            PointNetfeat(parameter, self.num_points, global_feat=True, trans=False),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, self.output_channels * self.num_points),
            nn.Tanh()
        )
        self.l_cd = self.configure_loss_function()

    def forward(self, x):
        x = x.permute(0, 2, 1)
        code = self.encoder(x)
        x = self.decoder(code)
        x = x.view(-1, self.output_channels, self.num_points)
        x = x.transpose(2, 1).contiguous()
        return x

    def backward_model(self, sparse_cloud, gt):
        loss = self.l_cd(sparse_cloud, gt)
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
        # scheduler_pct = torch.optim.lr_scheduler.StepLR(optimizer=optimizer_pct, step_size=40, gamma=0.1)
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
        # filename = "{}.pth.tar".format(filename)
        if os.path.isfile(filename):
            print("==> Loading from checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename)
            if checkpoint["model_ae_loss"] is not None:
                print("model_ae_loss: ", checkpoint["model_ae_loss"])
            if checkpoint["model_encoder_state"] is not None:
                self.encoder.load_state_dict(checkpoint["model_encoder_state"])
            if checkpoint["model_decoder_state"] is not None:
                self.decoder.load_state_dict(checkpoint["model_decoder_state"])
            if checkpoint["optimizer_pct_state"] is not None:
                self.model_optimizer["opt"].load_state_dict(checkpoint["optimizer_pct_state"])
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
            out_px1 = self(partial_point_cloud)
            gt1 = gt_point_cloud
            out_px1_generated = out_px1
            loss_px_gt, loss_cd = self.backward_model(sparse_cloud=out_px1_generated, gt=gt1)
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
                             loss_ae_updated, loss_cd_update, cd_sparse * 10000)
            if save_results and cd_sparse < 0.002:
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
            evaluate_class_choice_sparse["Plane"] * 10000, evaluate_class_choice_sparse["Cabinet"] * 10000,
            evaluate_class_choice_sparse["Car"] * 10000, evaluate_class_choice_sparse["Chair"] * 10000,
            evaluate_class_choice_sparse["Lamp"] * 10000, evaluate_class_choice_sparse["Couch"] * 10000,
            evaluate_class_choice_sparse["Table"] * 10000, evaluate_class_choice_sparse["Watercraft"] * 10000,
            sum(evaluate_loss_sparse) / len(evaluate_loss_sparse) * 10000)

        return sum(evaluate_loss_sparse) / len(evaluate_loss_sparse)

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
                out_px1 = self(partial_point_cloud)
                pc_point_completion_dense = out_px1
                if True:
                    for k in range(partial_point_cloud.shape[0]):
                        base_name = "kitti_car" + "_" + str(count_k) + "_pc_recon" + ".ply"
                        template_path = os.path.join(save_ply_path, base_name)
                        save_ply(pc=pc_point_completion_dense[k].cpu().detach().numpy(), path=template_path)

                        base_name = "kitti_car" + "_" + str(count_k) + "_pc" + ".ply"
                        template_path = os.path.join(save_ply_path, base_name)
                        save_ply(partial_point_cloud[k].cpu().detach().numpy(), path=template_path)
                        count_k += 1
