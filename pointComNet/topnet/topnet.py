import torch
import torch.nn as nn
import sys
from pathlib import Path

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))
from torch.autograd import Variable
from pointComNet.utils.loss_function import EMDLoss, CDLoss
from pointComNet.pytorch_utils.components.pytorch_base_model import BaseModel
from pointComNet.pytorch_utils.components.netUtils import NetUtil
import numpy as np
import math
import torch.optim as optim
from collections import OrderedDict
from pointComNet.pytorch_utils.components.Logger import *
from pointComNet.pytorch_utils.components.ioUtils import save_ply, make_dirs, copy_file
from pointComNet.pytorch_utils.components.dataSet import PointCompletionShapeNet

tree_arch = {}
tree_arch[2] = [32, 64]
tree_arch[4] = [4, 8, 8, 8]
tree_arch[6] = [2, 4, 4, 4, 4, 4]
tree_arch[8] = [2, 2, 2, 2, 2, 4, 4, 4]


def get_arch(nlevels, npts):
    logmult = int(math.log2(npts / 2048))
    assert 2048 * (2 ** (logmult)) == npts, "Number of points is %d, expected 2048x(2^n)" % (npts)
    arch = tree_arch[nlevels]
    while logmult > 0:
        last_min_pos = np.where(arch == np.min(arch))[0][-1]
        arch[last_min_pos] *= 2
        logmult -= 1
    return arch


class PointGenCon(nn.Module):
    def __init__(self, bottleneck_size=1024, output=3, use_tanh=True):
        super(PointGenCon, self).__init__()
        self.bottleneck_size = bottleneck_size
        channels = [self.bottleneck_size, self.bottleneck_size, self.bottleneck_size // 2, self.bottleneck_size // 4]
        # self.f_1 = nn.Sequential(nn.Conv1d(self.bottleneck_size, self.bottleneck_size, 1),
        #                          nn.BatchNorm1d(self.bottleneck_size),
        #                          nn.ReLU())
        #
        # self.f_2 = nn.Sequential(nn.Conv1d(self.bottleneck_size, self.bottleneck_size // 2, 1),
        #                          nn.BatchNorm1d(self.bottleneck_size // 2),
        #                          nn.ReLU())
        #
        # self.f_3 = nn.Sequential(nn.Conv1d(self.bottleneck_size // 2, self.bottleneck_size // 4, 1),
        #                          nn.BatchNorm1d(self.bottleneck_size // 4),
        #                          nn.ReLU())
        self.conv_module = nn.ModuleList([nn.Sequential(nn.Conv1d(channels[i - 1], channels[i], kernel_size=1),
                                                        nn.BatchNorm1d(channels[i]),
                                                        nn.ReLU()) for i in range(1, len(channels))])

        self.conv4 = torch.nn.Conv1d(self.bottleneck_size // 4, output, 1)
        self.use_tanh = use_tanh
        self.th = nn.Tanh()
        # self.bn1 = torch.nn.BatchNorm1d(self.bottleneck_size)
        # self.bn2 = torch.nn.BatchNorm1d(self.bottleneck_size // 2)
        # self.bn3 = torch.nn.BatchNorm1d(self.bottleneck_size // 4)

    def forward(self, x):
        for layer in self.conv_module:
            x = layer(x)
        # x = self.f_1(x)
        # x = self.f_2(x)
        # x = self.f_3(x)
        x = self.conv4(x)
        if self.use_tanh:
            x = self.th(x)
        return x


class TOPnetGen(nn.Module):
    def __init__(self, bottleneck=1024):
        super(TOPnetGen, self).__init__()
        # self.tree_arch = [2, 4, 4, 4, 4, 4]
        self.tree_arch = [2, 2, 2, 2, 2, 4, 4, 4, 4]
        self.Nout = [8, 8, 8, 8, 8, 8, 8, 8, 3]

        self.module = nn.ModuleList()
        self.tan = nn.Tanh()
        self.add_rand_size = 0
        self.use_rand = False
        print("--> use rand: ", self.use_rand)
        if self.use_rand:
            self.add_rand_size = 2
        self.bottleneck = bottleneck + self.add_rand_size
        self.level0 = NetUtil.SeqLinear(channels=[self.bottleneck, 256, 64, self.Nout[0] * self.tree_arch[0]])
        self.level_module = nn.ModuleList(
            [PointGenCon(bottleneck_size=self.Nout[i - 1] + self.bottleneck,
                         output=self.Nout[i] * self.tree_arch[i],
                         use_tanh=False) for i in
             range(1, len(self.tree_arch))])

    def forward(self, x):
        B, _, = x.shape
        x0 = x
        level_0_feature = self.tan(self.level0(x0)).reshape(-1, self.tree_arch[0], self.Nout[0])
        outs = [level_0_feature]
        for i, level in enumerate(self.level_module):
            inp = outs[-1]  # 32* 2*8 --> 32*8*8 --> 32*32*8
            x_rand = x
            y = x_rand.unsqueeze(dim=1)  # 32*1*1024 --> 32*1*1024 --> 32*1*1024
            y = y.repeat(1, inp.shape[1], 1)  # 32*2*1024 --> 32*8*1024 --> 32*32*1024
            y = torch.cat([inp, y], 2).transpose(2, 1)  # 32*1032*2 --> 32*1032*8 --> 32*1032*32
            y = level(y)  # 32*32*2 --> 32*32*8 --> 32*32*32
            y = y.reshape(B, -1, self.Nout[i + 1])  # 32*8*8  -> 32* 32*8 -> 32*128*8
            outs.append(self.tan(y))
        out_gen = outs[-1]
        return out_gen


class TopNet(nn.Module):
    def __init__(self, parameter):
        super(TopNet, self).__init__()
        self.encoder = None
        self.parameter = parameter
        self.tarch = get_arch(self.parameter["NLEVELS"], self.parameter["npts"])
        self.encoder_conv = NetUtil.SeqPointNetConv1d([3, 64, 128, self.parameter["code_nfts"]])
        self.encoder_Linear = nn.Linear(self.parameter["code_nfts"], self.parameter["code_nfts"])
        self.decoder = TOPnetGen(bottleneck=self.parameter["code_nfts"])

    def forward(self, x):
        x = x.permute(0, 2, 1)
        encoder_features = torch.max(self.encoder_conv(x), dim=-1)[0]
        encoder_features = self.encoder_Linear(encoder_features)
        out = self.decoder(encoder_features)
        return out
        # out_decoder = self.decoder_list[0]
        # out_decoder_list = [out_decoder]
        # for i in range(1, self.parameter["NLEVELS"]):
        #     y = encoder_features.unsqueeze(dim=-1)
        #     inp = out_decoder_list[-1]
        #     # y = torch.tile(y, [1, inp.shape[1], 1])
        #     y = y.repeat(1, inp.shape[1], 1)
        #     y = torch.cat([inp, y], 2)
        #     out_decoder_list.append(self.decoder_list[i](y))
        # return out_decoder_list[-1]

    # def create_decoder(self, code, nlevels):
    #     Nin = self.parameter["NFEAT"] + self.parameter["code_nfts"]
    #     Nout = self.parameter["NFEAT"]
    #     N0 = int(self.args.tarch[0])
    #     level0 = nn.Sequential(NetUtil.SeqLinear([256, 64, self.parameter["NFEAT"] * N0]), torch.nn.Tanh())
    #     self.decoder_list.append(level0)
    #     for i in range(1, nlevels):
    #         if i == nlevels - 1:
    #             Nout = 3
    #         self.decoder_list.append(self.create_level(i, Nin, Nout))
    #
    # def create_level(self, level, input_channels, output_channels):
    #     features = NetUtil.SeqPointNetConv1d([input_channels, int(input_channels / 2),
    #                                           int(input_channels / 4), int(input_channels / 8),
    #                                           output_channels * int(self.tarch[level])])
    #     level = nn.Sequential(features, torch.nn.Tanh())
    #     return level


class TopNetModel(BaseModel):
    def __init__(self, parameter, checkpoint_name, best_name, checkpoint_path, logger_file_name):
        super(TopNetModel, self).__init__(parameter, checkpoint_name, best_name, checkpoint_path, logger_file_name)
        self.model = TopNet(parameter=parameter)
        self.l_cd = self.configure_loss_function()

    def forward(self, x):
        return self.model(x)

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
            if checkpoint["model_state"] is not None:
                self.model.load_state_dict(checkpoint["model_state"])
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
            "model_ae_loss": loss_cd,
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
                             loss_ae_updated, loss_cd_update, cd_sparse * 10000)
            if save_results and cd_sparse <= 0.003:
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
        if count_k >= 100:
            return
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
