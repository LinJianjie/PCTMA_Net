import sys
from pathlib import Path

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[2]
sys.path.append(str(root))

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from pointComNet.atlasNet.common import PointNetfeat, weights_init
from pointComNet.utils.loss_function import EMDLoss, CDLoss
import torch.optim as optim
from collections import OrderedDict
import os
import time
from pointComNet.pytorch_utils.components.Logger import *
from pointComNet.pytorch_utils.components.ioUtils import save_ply, make_dirs, copy_file
from pointComNet.pytorch_utils.components.dataSet import PointCompletionShapeNet


def AtlasNet_setup(args):
    args.odir = 'results/%s/AtlasNet_%s' % (args.dataset, args.dist_fun)
    grain = int(np.sqrt(args.npts / args.nb_primitives))
    grain = grain * 1.0
    n = ((grain + 1) * (grain + 1) * args.nb_primitives)
    if n < args.npts:
        grain += 1
    args.npts = (grain + 1) * (grain + 1) * args.nb_primitives
    args.odir += '_npts%d' % (args.npts)
    args.odir += '_NBP%d' % (args.nb_primitives)
    args.odir += '_lr%.4f' % (args.lr)
    args.odir += '_' + args.optim
    args.odir += '_B%d' % (args.batch_size)
    args.classmap = ''

    # generate regular grid
    vertices = []
    for i in range(0, int(grain + 1)):
        for j in range(0, int(grain + 1)):
            vertices.append([i / grain, j / grain])

    grid = [vertices for i in range(0, args.nb_primitives)]
    print("grain", grain, 'number vertices', len(vertices) * args.nb_primitives)
    args.grid = grid


def AtlasNet_create_model(args):
    """ Creates model """
    model = nn.DataParallel(AtlasNet(args, num_points=args.npts, nb_primitives=args.nb_primitives))
    args.enc_params = sum([p.numel() for p in model.module.encoder.parameters()])
    args.dec_params = sum([p.numel() for p in model.module.decoder.parameters()])
    args.nparams = sum([p.numel() for p in model.module.parameters()])
    print('Total number of parameters: {}'.format(args.nparams))
    print(model)
    model.cuda()
    model.apply(weights_init)
    return model


def AtlasNet_step(args, targets_in, clouds_data):
    targets = Variable(torch.from_numpy(targets_in), requires_grad=False).float().cuda()
    targets = targets.transpose(2, 1).contiguous()
    inp = Variable(torch.from_numpy(clouds_data[1]), requires_grad=False).float().cuda()
    outputs = args.model.forward(inp, args.grid)
    targets = targets.transpose(2, 1).contiguous()
    N = targets.size()[1]
    dist1, dist2 = eval(args.dist_fun)()(outputs, targets)
    # EMD not working in pytorch (see pytorch-setup.md)
    # emd_cost = args.emd_mod(outputs[:, 0:N,:], targets)/N
    # emd_cost = emd_cost.data.cpu().numpy()
    emd_cost = 0  # args.emd_mod(outputs[:, 0:N, :], targets)/N
    emd_cost = np.array([0] * args.batch_size)  # emd_cost.data.cpu().numpy()

    loss = torch.mean(dist2) + torch.mean(dist1)
    dist1 = dist1.data.cpu().numpy()
    dist2 = dist2.data.cpu().numpy()

    if args.model.training:
        return loss, dist1, dist2, emd_cost, outputs.data.cpu().numpy()
    else:
        return loss.item(), dist1, dist2, emd_cost, outputs.data.cpu().numpy()


class PointGenCon(nn.Module):
    def __init__(self, bottleneck_size=2500):
        self.bottleneck_size = bottleneck_size
        super(PointGenCon, self).__init__()
        self.conv1 = torch.nn.Conv1d(self.bottleneck_size, self.bottleneck_size, 1)
        self.conv2 = torch.nn.Conv1d(self.bottleneck_size, int(self.bottleneck_size / 2), 1)
        self.conv3 = torch.nn.Conv1d(int(self.bottleneck_size / 2), int(self.bottleneck_size / 4), 1)
        self.conv4 = torch.nn.Conv1d(int(self.bottleneck_size / 4), 3, 1)

        self.th = nn.Tanh()
        self.bn1 = torch.nn.BatchNorm1d(self.bottleneck_size)
        self.bn2 = torch.nn.BatchNorm1d(int(self.bottleneck_size / 2))
        self.bn3 = torch.nn.BatchNorm1d(int(self.bottleneck_size / 4))

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.th(self.conv4(x))
        return x


class AtlasNet(nn.Module):
    def __init__(self, parameter, checkpoint_name="ckpt", best_name="best", logger_file_name="log.log",
                 checkpoint_path=None):
        super(AtlasNet, self).__init__()
        self.checkpoint_name = checkpoint_name
        self.checkpoint_path = checkpoint_path
        self.best_name = best_name
        self.parameter = parameter
        self.num_points = parameter["num_points"]
        self.bottleneck_size = parameter["bottleneck_size"]
        self.nb_primitives = parameter["nb_primitives"]
        self.time_str = time.strftime("%Y%m%d_%H%M%S")
        logger_file_name = logger_file_name + "_" + self.time_str + ".log"
        self.Logger = Logger(filename=logger_file_name)

        self.encoder = nn.Sequential(
            PointNetfeat(parameter, self.num_points, global_feat=True, trans=False),
            nn.Linear(1024, self.bottleneck_size),
            nn.BatchNorm1d(self.bottleneck_size),
            nn.ReLU()
        )
        self.decoder = nn.ModuleList(
            [PointGenCon(bottleneck_size=2 + self.bottleneck_size) for i in range(0, self.nb_primitives)])

        self.l_cd = self.configure_loss_function()
        self.optimizer_pct = None

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.encoder(x)
        outs = []
        for i in range(0, self.nb_primitives):
            rand_grid = Variable(torch.cuda.FloatTensor(x.size(0), 2, self.num_points // self.nb_primitives))
            rand_grid.data.normal_(0, 1)
            rand_grid = rand_grid / torch.sqrt(torch.sum(rand_grid ** 2, dim=1, keepdim=True))
            y = x.unsqueeze(2).expand(x.size(0), x.size(1), rand_grid.size(2)).contiguous()
            y = torch.cat((rand_grid, y), 1).contiguous()
            outs.append(self.decoder[i](y))
        return torch.cat(outs, 2).contiguous().transpose(2, 1).contiguous()

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

    def configure_optimizers(self):
        optimizer_pct = optim.Adam(self.parameters(), lr=1e-3, betas=(0.9, 0.999))
        scheduler_pct = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_pct, self.parameter["epochs"],
                                                                   eta_min=1e-3)
        # scheduler_pct = torch.optim.lr_scheduler.StepLR(optimizer=optimizer_pct, step_size=40, gamma=0.1)
        opt_pct = OrderedDict({"opt": optimizer_pct, "scheduler": scheduler_pct})
        self.optimizer_pct = opt_pct
        return opt_pct

    def backward_atlastNet(self, gt1=None, pc1=None):
        cd_1 = self.l_cd(pc1, gt1)
        loss = cd_1
        return loss, cd_1

    def backward_atlastNet_update(self, loss_pct):
        self.optimizer_pct["opt"].zero_grad()
        loss_pct.backward()
        self.optimizer_pct["opt"].step()
        self.optimizer_pct["scheduler"].step()

    def save_checkpoint(self, state, best_model_name):
        # time_str = time.strftime("%Y%m%d_%H%M%S")
        best_model = self.time_str + "_" + best_model_name
        print("best_model: ", best_model)
        self.best_name = best_model
        save_path = os.path.join(self.checkpoint_path, best_model)
        torch.save(state, save_path)

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
                self.optimizer_pct["opt"].load_state_dict(checkpoint["optimizer_pct_state"])
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
            loss_pct = None
            loss_gt = None
            loss_px = None
            loss_cd = None
            loss_px_gt = None
            # if self.train_gt:
            #     out_gt_3, out_gt_2, out_gt_1 = self(gt_point_cloud)
            #     loss_gt, loss_cd = self.backward_pct()
            #     loss_pct = loss_gt
            # if self.train_pt:
            #     out_px = self(partial_point_cloud)
            #     loss_px, loss_cd = self.backward_pct(output_decode=out_px, target=partial_point_cloud)
            #     loss_pct = loss_px
            # if self.train_pt_to_gt:
            out_px1 = self(partial_point_cloud)
            gt1 = gt_point_cloud
            # gt2, _, _ = farthest_point_sampling(x=gt1, ratio=0.5)
            # gt3, _, _ = farthest_point_sampling(x=gt2, ratio=0.25)
            # px_fps, _, _ = farthest_point_sampling(partial_point_cloud, ratio=0.5)
            # out_px1_fps, _, _ = farthest_point_sampling(out_px1, ratio=0.5)
            # out_px1_generated = torch.cat([partial_point_cloud, out_px1], dim=1)
            out_px1_generated = out_px1

            # out_px1_expand = torch.cat([out_px1, partial_point_cloud], dim=1)
            # out_px1_fps, _, _ = farthest_point_sampling(out_px1_expand, ratio=0.5)
            loss_px_gt, loss_cd = self.backward_atlastNet(gt1=gt1, pc1=out_px1_generated)
            loss_pct = loss_px_gt
            # if self.train_gt and self.train_pt:
            #     loss_pct = (loss_gt + loss_px) / 2.0
            # loss_pct = (loss_px + loss_gt) / 2.0
            self.backward_atlastNet_update(loss_pct=loss_pct)
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
            "optimizer_pct_state": self.optimizer_pct["opt"].state_dict()
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
            if loss_cd_update < loss_cd and loss_cd_update < 0.004 and save_results:
                loss_cd = loss_cd_update
                save_results = False
                state = {
                    "batch_size": batch_size,
                    "model_ae_loss": loss_cd,
                    "model_encoder_state": self.encoder.state_dict(),
                    "model_decoder_state": self.decoder.state_dict(),
                    "optimizer_pct_state": self.optimizer_pct["opt"].state_dict()
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
