import open3d
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np
import time
from pointComNet.utils.loss_function import EMDLoss, CDLoss
import numpy as np
from pointComNet.pytorch_utils.components.pytorch_base_model import BaseModel
import torch.optim as optim
from collections import OrderedDict
from pointComNet.pytorch_utils.components.Logger import *
from pointComNet.pytorch_utils.components.ioUtils import save_ply, make_dirs, copy_file
from pointComNet.pytorch_utils.components.dataSet import PointCompletionShapeNet


class PCN_Encoder(nn.Module):
    def __init__(self):
        super(PCN_Encoder, self).__init__()
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
        features = self.first_mpl_last(features)  # 32*256*2048
        features_global = torch.max(features, dim=-1, keepdim=True)[0]  # 32*256*1
        features_global = features_global.repeat(1, 1, xyz.shape[2])  # 32*256*2048
        features = torch.cat([features, features_global], dim=1)  # 32*512*2048
        features = self.second_mpl(features)
        features = self.second_mpl_last(features)
        features = torch.max(features, dim=-1)[0]
        return features


class PCN_Decoder(nn.Module):
    def __init__(self):
        super(PCN_Decoder, self).__init__()
        # self.num_coarse = 1024
        # self.grid_size = 4
        self.grid_scale = 0.05
        self.npts = 16384
        grid_size = {2048: 2, 4096: 2, 8192: 4, 16384: 4}
        self.grid_size = grid_size[self.npts]
        self.num_coarse = int(self.npts / (self.grid_size ** 2))
        self.npts = (self.grid_size ** 2) * self.num_coarse

        self.num_fine = self.npts
        coarse1 = [1024, 1024, 1024]

        self.decoder = nn.Sequential(*[nn.Sequential(torch.nn.Linear(coarse1[i - 1], coarse1[i]),
                                                     torch.nn.ReLU()) for i in range(1, len(coarse1))])
        self.decoder_last = torch.nn.Linear(1024, self.num_coarse * 3)

        mlpsfold = [self.num_coarse + 2 + 3, 512, 512]
        self.fold_mpl = nn.Sequential(*[nn.Sequential(torch.nn.Conv1d(mlpsfold[i - 1], mlpsfold[i], kernel_size=1),
                                                      torch.nn.BatchNorm1d(mlpsfold[i]),
                                                      torch.nn.ReLU()) for i in range(1, len(mlpsfold))])
        self.fold_mpl_last = torch.nn.Conv1d(512, 3, kernel_size=1)

    def forward(self, features):
        coarse = self.decoder(features)
        coarse = self.decoder_last(coarse)
        coarse = coarse.view(-1, self.num_coarse, 3)
        ##FOLDING
        grid_row = torch.linspace(-0.05, 0.05, self.grid_size).cuda()
        grid_column = torch.linspace(-0.05, 0.05, self.grid_size).cuda()
        grid = torch.meshgrid(grid_row, grid_column)
        grid = torch.reshape(torch.stack(grid, dim=2), (-1, 2)).unsqueeze(0)
        grid_feat = grid.repeat([features.shape[0], self.num_coarse, 1])
        point_feat = coarse.unsqueeze(2).repeat([1, 1, self.num_fine // self.num_coarse, 1])
        point_feat = torch.reshape(point_feat, [-1, self.num_fine, 3])
        global_feat = features.unsqueeze(1).repeat([1, self.num_fine, 1])
        feat = torch.cat([grid_feat, point_feat, global_feat], dim=2)
        center = coarse.unsqueeze(2).repeat([1, 1, self.num_fine // self.num_coarse, 1])
        center = center.reshape(-1, 3, self.num_fine)
        fine = self.fold_mpl(feat.permute(0, 2, 1))
        fine = self.fold_mpl_last(fine)
        fine = fine + center
        return coarse, fine.transpose(2, 1)


# class PCNEMD(nn.Module):
#     def __init__(self):
#         super(PCNEMD, self).__init__()
#         self.num_coarse = 1024
#         self.grid_size = 4
#         self.grid_scale = 0.05
#         self.num_fine = self.grid_size ** 2 * self.num_coarse
#         self.npts = [1]
#         # alpha  = [10000, 20000, 50000],[0.01, 0.1, 0.5, 1.0]
#         #### ENCODER
#
#         ## first mlp
#         mlps1 = [3, 128, 256]
#         # for m in range(0, len(mlps1) - 1):
#         #     first_mlp_list.append(nn.Conv1d(in_features, mlps1[m], 1))
#         #     first_mlp_list.append(nn.ReLU())
#         #     in_features = mlps1[m]
#         # first_mlp_list.append(nn.Conv1d(in_features, mlps1[-1], 1))
#         # self.first_mpl = nn.Sequential(*first_mlp_list)
#         self.first_mpl = nn.Sequential(*[nn.Sequential(torch.nn.Conv1d(mlps1[i - 1], mlps1[i], kernel_size=1),
#                                                        torch.nn.BatchNorm1d(mlps1[i]),
#                                                        torch.nn.ReLU()) for i in range(1, len(mlps1))])
#
#         ## Second  mlp
#         mlps2 = [mlps1[-1] * 2, 512, 1024]
#         self.second_mpl = nn.Sequential(*[nn.Sequential(torch.nn.Conv1d(mlps2[i - 1], mlps2[i], kernel_size=1),
#                                                         torch.nn.BatchNorm1d(mlps2[i]),
#                                                         torch.nn.ReLU()) for i in range(1, len(mlps2))])
#         # mlps2 = [512, 1024]
#         # second_mlp_list = []
#         # in_features = 512
#         # for m in range(0, len(mlps2) - 1):
#         #     second_mlp_list.append(nn.Conv1d(in_features, mlps2[m], 1))
#         #     second_mlp_list.append(nn.ReLU())
#         #     in_features = mlps2[m]
#         # second_mlp_list.append(nn.Conv1d(in_features, mlps2[-1], 1))
#         # self.second_mpl = nn.Sequential(*second_mlp_list)
#
#         #### DECODER
#         coarse1 = [1024, 1024, 1024, self.num_coarse * 3]
#
#         self.decoder = nn.Sequential(*[nn.Sequential(torch.nn.Linear(coarse1[i - 1], coarse1[i]),
#                                                      torch.nn.ReLU()) for i in range(1, len(coarse1))])
#         # decoder_list = []
#         # for m in range(0, len(coarse1) - 1):
#         #     decoder_list.append(nn.Linear(in_features, coarse1[m]))
#         #     in_features = coarse1[m]
#         # decoder_list.append(nn.Linear(in_features, coarse1[-1]))
#         # self.decoder = nn.Sequential(*decoder_list)
#
#         ## FOLDING
#         mlpsfold = [self.num_coarse + 2 + 3, 512, 512, 3]
#         self.fold_mpl = nn.Sequential(*[nn.Sequential(torch.nn.Conv1d(mlpsfold[i - 1], mlpsfold[i], kernel_size=1),
#                                                       torch.nn.BatchNorm1d(mlpsfold[i]),
#                                                       torch.nn.ReLU()) for i in range(1, len(mlpsfold))])
#         #
#         # for m in range(0, len(mlpsfold) - 1):
#         #     fold_mlp_list.append(nn.Conv1d(in_features, mlpsfold[m], 1))
#         #     fold_mlp_list.append(nn.ReLU())
#         #     in_features = mlpsfold[m]
#         # fold_mlp_list.append(nn.Conv1d(in_features, mlpsfold[-1], 1))
#         # self.fold_mpl = nn.Sequential(*fold_mlp_list)
#
#     # def point_maxpool(self, features, npts, keepdims=True):
#     #     splitted = torch.split(features, npts[0], dim=1)
#     #     outputs = [torch.max(f, dim=2, keepdims=keepdims)[0] for f in splitted]
#     #     return torch.cat(outputs, dim=0)
#     #     # return torch.max(features, dim=2, keepdims=keepdims)[0]
#     #
#     # def point_unpool(self, features, npts):
#     #     features = torch.split(features, features.shape[0], dim=0)
#     #     outputs = [f.repeat([1, npts[i], 1]) for i, f in enumerate(features)]
#     #     return torch.cat(outputs, dim=1)
#     #     # return features.repeat([1, 1, 256])
#
#     def forward(self, xyz):
#         xyz = xyz.permute(0, 2, 1)
#         #####ENCODER
#         features = self.first_mpl(xyz)
#         # print("features: ", features.shape)
#         features_global = torch.max(features, dim=1, keepdim=True)[0]
#         # features_global = self.point_maxpool(features, self.npts, keepdims=True)
#         # print("feature_global 1: ", features_global.shape)
#         # features = tf.concat([features, tf.tile(features_global, [1, tf.shape(inputs)[1], 1])], axis=2)
#         # features_global = self.point_unpool(features_global, self.npts)
#         features_global = features_global.repeat(1, features.shape[1], 1)
#         # print("feature_global 2: ", features_global.shape)
#         features = torch.cat([features, features_global], dim=1)
#
#         # print("features_global: ", features_global.shape, " features: ", features.shape)
#         # features = torch.cat([features, features_global.permute(0, 2, 1)], dim=1)
#         features = self.second_mpl(features)
#         # features = self.point_maxpool(features, self.npts).squeeze(2)
#         features = torch.max(features, dim=-1)[0]
#         # features = torch.max(features, dim=-1)[0]
#         ##DECODER
#         coarse = self.decoder(features)
#         # print("coarse: ", coarse.shape)
#         coarse = coarse.view(-1, self.num_coarse, 3)
#         # print("after viewer coarse: ", coarse.shape)
#         ##FOLDING
#         grid_row = torch.linspace(-0.05, 0.05, self.grid_size).cuda()
#         grid_column = torch.linspace(-0.05, 0.05, self.grid_size).cuda()
#         grid = torch.meshgrid(grid_row, grid_column)
#         grid = torch.reshape(torch.stack(grid, dim=2), (-1, 2)).unsqueeze(0)
#         grid_feat = grid.repeat([features.shape[0], 1024, 1])
#         point_feat = coarse.unsqueeze(2).repeat([1, 1, self.num_fine // self.num_coarse, 1])
#         point_feat = torch.reshape(point_feat, [-1, self.num_fine, 3])
#         # print("point_feat: ", point_feat.shape)
#         global_feat = features.unsqueeze(1).repeat([1, self.num_fine, 1])
#         # print("global_feat: ", global_feat.shape)
#         feat = torch.cat([grid_feat, point_feat, global_feat], dim=2)
#         # print("feat: ", feat.shape)
#         center = coarse.unsqueeze(2).repeat([1, 1, self.num_fine // self.num_coarse, 1])
#         center = torch.reshape(center, [-1, self.num_fine, 3])
#         fine = self.fold_mpl(feat.permute(0, 2, 1))
#         fine = fine.permute(0, 2, 1) + center
#         return coarse, fine


class PCNModel(nn.Module):
    def __init__(self, parameter, checkpoint_name, best_name, checkpoint_path, logger_file_name):
        super(PCNModel, self).__init__()
        self.device = None
        self.model_optimizer = None
        self.checkpoint_name = checkpoint_name
        self.checkpoint_path = checkpoint_path
        self.best_name = best_name
        self.parameter = parameter
        self.model = None
        self.time_str = time.strftime("%Y%m%d_%H%M%S")
        logger_file_name = logger_file_name + "_" + self.time_str + ".log"
        self.Logger = Logger(filename=logger_file_name)

        self.encoder = PCN_Encoder()
        self.decoder = PCN_Decoder()
        self.l_cd, self.l_emd = self.configure_loss_function()

    def forward(self, xyz):
        features = self.encoder(xyz)
        coarse, fine = self.decoder(features)
        return coarse, fine

    def backward_model(self, coarse, fine, gt):
        loss_coarse = self.l_cd(coarse, gt)
        loss_fine = self.l_cd(fine, gt)
        loss = loss_coarse + loss_fine
        return loss, loss_fine

    def backward_model_update(self, loss):
        self.model_optimizer["opt"].zero_grad()
        loss.backward()
        self.model_optimizer["opt"].step()
        self.model_optimizer["scheduler"].step()

    def save_checkpoint(self, state, best_model_name):
        # time_str = time.strftime("%Y%m%d_%H%M%S")
        best_model = self.time_str + "_" + best_model_name
        print("best_model: ", best_model)
        self.best_name = best_model
        save_path = os.path.join(self.checkpoint_path, best_model)
        torch.save(state, save_path)

    def configure_optimizers(self):
        optimizer_pct = optim.Adam(self.parameters(), lr=1e-3, betas=(0.9, 0.999))
        scheduler_pct = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_pct, self.parameter["epochs"],
                                                                   eta_min=1e-3)
        opt_pct = OrderedDict({"opt": optimizer_pct, "scheduler": scheduler_pct})
        self.model_optimizer = opt_pct
        return opt_pct

    @staticmethod
    def configure_loss_function():
        pointCloud_cd = CDLoss()
        pointCloud_emd = EMDLoss()
        return pointCloud_cd, pointCloud_emd

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
            coarse, fine = self(partial_point_cloud)
            gt1 = gt_point_cloud
            loss_px_gt, loss_cd = self.backward_model(coarse=coarse, fine=fine, gt=gt1)
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
            self.Logger.INFO('Train %d/%d, loss_ae: %.4f, loss_cd: %.4f, best cd: %.4f \n', at_epoch, n_epochs,
                             loss_ae_updated, loss_cd_update, cd_sparse * 10000)
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
                coarse, fine = self(partial_point_cloud)
                out_px1 = fine
                cd = self.l_cd(fine, gt_point_cloud)
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
            '====> cd_sparse: Airplane: %.4f, Cabinet: %.6f, Car: %.4f, Chair: %.4f, Lamp: %.4f, Sofa: %.4f, Table: %.4f, Watercraft: %.4f, mean: %.4f',
            evaluate_class_choice_sparse["Plane"] * 10000, evaluate_class_choice_sparse["Cabinet"] * 10000,
            evaluate_class_choice_sparse["Car"] * 10000, evaluate_class_choice_sparse["Chair"] * 10000,
            evaluate_class_choice_sparse["Lamp"] * 10000, evaluate_class_choice_sparse["Couch"] * 10000,
            evaluate_class_choice_sparse["Table"] * 10000, evaluate_class_choice_sparse["Watercraft"] * 10000,
            sum(evaluate_loss_sparse) / len(evaluate_loss_sparse) * 10000)

        return sum(evaluate_loss_sparse) / len(evaluate_loss_sparse)


if __name__ == '__main__':
    # alpha [ 0.01,0.1,0.5,1.0]
    pass
