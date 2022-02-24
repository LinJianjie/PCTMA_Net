import sys
from pathlib import Path

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))
from pointComNet.pytorch_utils.components.transformerNet.transformer_model import NetTEncoder
import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict
from pointComNet.utils.loss_function import EMDLoss, CDLoss, CDLossEval
import time
import numpy as np
from pointComNet.pytorch_utils.components.Logger import *
from pointComNet.pctma.autoencoder import *
from pointComNet.pytorch_utils.components.ioUtils import save_ply, make_dirs, copy_file
from pointComNet.pytorch_utils.components.torch_cluster_sampling import farthest_point_sampling, K_NN, \
    RandomPointSampling
from pointComNet.pytorch_utils.components.dataSet import PointCompletionShapeNet


class PCTMA_Net(nn.Module):
    def __init__(self, parameter, checkpoint_name="ckpt", best_name="best", logger_file_name="log.log",
                 checkpoint_path=None):
        super(PCTMA_Net, self).__init__()
        self.device = None
        self.optimizer_pct = None
        local_attention_size = None
        self.checkpoint_name = checkpoint_name
        self.checkpoint_path = checkpoint_path
        self.best_name = best_name
        self.time_str = time.strftime("%Y%m%d_%H%M%S")
        logger_file_name = logger_file_name + "_" + self.time_str + "_ppd" + ".log"
        self.Logger = Logger(filename=logger_file_name)
        self.parameter = parameter
        self.net_pcT = NetTEncoder(src_vocab=3,
                                   local_attention_size=local_attention_size,
                                   N=self.parameter["Num_Encoder"],
                                   d_model=self.parameter["d_model"],
                                   d_ff=self.parameter["d_ff"],
                                   num_head=self.parameter["num_head"],
                                   dropout=self.parameter["dropout"],
                                   use_cmlp=self.parameter["use_cmlp"],
                                   channels=parameter["En_channels"])
        self.netDe = NetMADecoder(args=parameter, num_points=parameter["De_numpoints"],
                                  channels=self.net_pcT.last_layer_size)
        self.point_sampling = RandomPointSampling(n_points=2048)
        self.l_cd, self.l_emd, self.l_consistence = self.configure_loss_function()
        self.l_cd_eval = CDLossEval()
        self.train_pt = self.parameter["train_pt"]
        self.train_gt = self.parameter["train_gt"]
        self.train_pt_to_gt = self.parameter["train_pt_to_gt"]
        self.use_emd = self.parameter["use_emd"]

        self.Logger.INFO(
            "---> use_cd: %r, use_emd: %r, combined_pc: %r, ppd_loss: %r, down_sampling: %r, use_consistence: %r, n_primitives: %r, use_atlas: %r, gen_file: %r",
            self.parameter["use_cd"],
            self.parameter["use_emd"],
            self.parameter["combined_pc"],
            self.parameter["ppd_loss"],
            self.parameter["down_sampling"],
            self.parameter["use_consistence"],
            self.parameter["n_primitives"],
            self.parameter["use_atlas"],
            self.parameter["gene_file"],
        )

    def forward(self, src_input, gt=False):
        z_hat = self.net_pcT(src_input)
        if gt:
            return z_hat
        else:
            pc3_xyz, pc2_xyz, pc1_xyz = self.netDe(z_hat)
            return pc3_xyz, pc2_xyz, pc1_xyz, z_hat

    def point_loss(self, generated, gt):
        cd_loss = self.l_cd(gt, generated)
        if self.parameter["use_emd"]:
            emd_loss = self.l_emd(gt, generated)
        if self.parameter["use_cd"] and self.parameter["use_emd"]:
            loss = cd_loss * self.parameter["w_cd"] + emd_loss * self.parameter["w_emd"]
        if self.parameter["use_cd"] and not self.parameter["use_emd"]:
            loss = cd_loss
        if not self.parameter["use_cd"] and self.parameter["use_emd"]:
            loss = emd_loss
        return loss, cd_loss

    def backward_emd_loss(self, gt, pc):
        loss = []
        cd = []
        for i in range(len(pc)):
            loss_1, cd_1 = self.point_loss(generated=pc[i], gt=gt)
            loss.append(loss_1)
            cd.append(cd_1)
        loss = sum(loss) / len(pc)
        cd = sum(cd) / len(pc)
        return loss, cd

    def backward_compute_loss(self, gt3=None, gt2=None, gt1=None, pc3=None, pc2=None, pc1=None):

        loss_1, cd_1 = self.point_loss(pc1, gt1)
        loss = loss_1

        if gt2 is not None and pc2 is not None:
            loss_2, cd_2 = self.point_loss(pc2, gt2)
            loss += loss_2 * 1

        if gt3 is not None and pc3 is not None:
            loss_3, cd_3 = self.point_loss(pc3, gt3)
            loss += loss_3 * 1

        return loss, cd_1

    def backward_loss_update(self, loss_pct):
        self.optimizer_pct["opt"].zero_grad()
        loss_pct.backward()
        self.optimizer_pct["opt"].step()
        self.optimizer_pct["scheduler"].step()

    @staticmethod
    def configure_loss_function():
        pointCloud_reconstruction_loss = CDLoss()
        emd_loss = EMDLoss()
        consist_loss = nn.CosineSimilarity(dim=1, eps=1e-6)
        return pointCloud_reconstruction_loss, emd_loss, consist_loss

    def toCuda(self, device):
        self.device = device
        if torch.cuda.device_count() > 1:
            print("====> use data parallel")
            self.net_pcT = nn.DataParallel(self.net_pcT)
            self.netDe = nn.DataParallel(self.netDe)
            self.net_pcT.to(device)
            self.netDe.to(device)
            # self.l_cd.to(device)
            # self.l_emd.to(device)
        else:
            print("====> use only one cuda")
            self.net_pcT.to(device)
            self.netDe.to(device)
            # self.l_cd.to(device)
            # self.l_emd.to(device)

    def configure_optimizers(self):
        optimizer_pct = optim.Adam(self.parameters(), lr=1e-3, betas=(0.9, 0.999))
        scheduler_pct = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_pct, self.parameter["epochs"],
                                                                   eta_min=1e-3)
        opt_pct = OrderedDict({"opt": optimizer_pct, "scheduler": scheduler_pct})
        self.optimizer_pct = opt_pct
        return opt_pct

    def save_config(self):
        filename = "pct.yaml"
        config_name = self.time_str + "_" + filename
        source = os.path.join(os.path.dirname(__file__), "../config", filename)
        destination = self.checkpoint_path
        copy_file(source=source, des=destination, filename=filename, rename=config_name)

    def save_checkpoint(self, state, best_model_name):
        best_model = self.time_str + "_" + best_model_name
        print("---> save best_model: ", best_model)
        self.best_name = best_model
        save_path = os.path.join(self.checkpoint_path, best_model)
        torch.save(state, save_path)
        self.save_config()

    def load_checkpoint(self, filename="checkpoint"):
        if os.path.isfile(filename):
            print("==> Loading from checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename)
            if checkpoint["model_ae_loss"] is not None:
                print("model_ae_loss: ", checkpoint["model_ae_loss"])
            if checkpoint["model_pct_state"] is not None:
                self.net_pcT.load_state_dict(checkpoint["model_pct_state"])
            if checkpoint["model_De_state"] is not None:
                self.netDe.load_state_dict(checkpoint["model_De_state"])
            if checkpoint["optimizer_pct_state"] is not None:
                self.optimizer_pct["opt"].load_state_dict(checkpoint["optimizer_pct_state"])
            print("==> Done")
            return None
        else:
            print("==> Checkpoint '{}' not found".format(filename))
            return None

    def count_parameters(self):
        number_of_parameter = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("the number of parameter: ", number_of_parameter)

    def train_one_epoch(self, train_loader, at_epoch, n_epochs, batch_size):
        count_ae = 0
        train_loss_ae = 0
        train_loss_cd = 0
        self.train()
        for i, dataset in enumerate(train_loader):
            gt_point_cloud, partial_point_cloud, label_point_cloud = dataset
            gt_point_cloud = gt_point_cloud.to(self.device)
            partial_point_cloud = partial_point_cloud.to(self.device)
            out_px3, out_px2, out_px1, z_hat_pt = self(partial_point_cloud)
            gt1 = gt_point_cloud
            if self.parameter["combined_pc"]:
                pc_point_completion_dense = torch.cat([partial_point_cloud, out_px1], dim=1)
                # if self.parameter["use_emd"]:
                #     loss_px_gt_dense = None
                #     partial_point_cloud_sparse = []
                #     down_ratio = 2048 / pc_point_completion_dense.shape[1]
                #     repate = pc_point_completion_dense.shape[1] // 2048
                #     for i in range(repate):
                #         partial_point_cloud_sparse1, _, _ = farthest_point_sampling(pc_point_completion_dense,
                #                                                                     ratio=down_ratio)
                #         partial_point_cloud_sparse.append(partial_point_cloud_sparse1)
                #
                # if self.parameter["use_emd"] and not self.parameter["use_cd"]:
                #     loss_px_gt_dense, loss_cd = self.backward_emd_loss(gt=gt1, pc=partial_point_cloud_sparse)
                # else:
                loss_px_gt_dense, loss_cd = self.backward_compute_loss(gt1=gt1, pc1=pc_point_completion_dense)
                loss_px_gt = loss_px_gt_dense
            loss_pct = loss_px_gt
            # if self.train_gt and self.train_pt:
            #     loss_pct = (loss_gt + loss_px) / 2.0
            # loss_pct = (loss_px + loss_gt) / 2.0
            self.backward_loss_update(loss_pct=loss_pct)
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
            "model_pct_state": self.net_pcT.state_dict(),
            "model_De_state": self.netDe.state_dict(),
            "optimizer_pct_state": self.optimizer_pct["opt"].state_dict()
        }
        cd_sparse = 1000
        cd_dense = 1000
        save_checkout = False
        cd_sparse_best = 1000
        cd_dense_best = 1000
        for at_epoch in range(start_epoch, n_epochs):
            loss_ae_updated, loss_cd_update = self.train_one_epoch(train_loader=train_loader,
                                                                   at_epoch=at_epoch,
                                                                   n_epochs=n_epochs,
                                                                   batch_size=batch_size)
            cd_sparse_update, cd_dense_update = self.evaluation_step(test_loader=test_loader)
            if cd_dense_update < cd_dense:
                # print('the mean cd_loss_sparse: {:.6f}, cd_loss_dense: {:.6f}, best: {:.6f}'.format(
                #     cd_sparse_update, cd_dense_update, cd_dense))
                # cd_dense_best = np.min(cd_dense_update, cd_dense)
                # cd_sparse_best = np.min(cd_sparse_update, cd_sparse)
                if cd_dense_best > cd_dense_update:
                    cd_dense_best = cd_dense_update
                    cd_sparse_best = cd_sparse_update
                else:
                    cd_dense_best = cd_dense
                    cd_sparse_best = cd_sparse

                cd_dense = cd_dense_best
                cd_sparse = cd_sparse_best
                if cd_dense < 0.0013:
                    # cd_sparse_update, cd_dense_update = self.evaluation_step(test_loader=test_loader)
                    save_checkout = True

            self.Logger.INFO(
                'Train %d/%d, loss_ae: %.6f, loss_cd: %.6f, evaluate_cd_spares: %.4f, evaluate_cd_dense: %.4f \n',
                at_epoch, n_epochs, loss_ae_updated, loss_cd_update * 10000, cd_sparse * 10000, cd_dense * 10000)

            if loss_cd_update < loss_cd and loss_cd_update < 0.0013 and save_checkout:
                save_checkout = False
                loss_cd = loss_cd_update
                state = {
                    "batch_size": batch_size,
                    "model_ae_loss": cd_sparse,
                    "model_pct_state": self.net_pcT.state_dict(),
                    "model_De_state": self.netDe.state_dict(),
                    "optimizer_pct_state": self.optimizer_pct["opt"].state_dict()
                }
                self.save_checkpoint(state=state, best_model_name=best_model_name)

    def evaluation_step(self, test_loader, check_point_name=None):
        # self.count_parameters()
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

                out_px3, out_px2, out_px1, _ = self(partial_point_cloud)
                # for k in range(partial_point_cloud.shape[0]):
                #     if test_loader.dataset.label_to_category(label_point_cloud[k]) == "plane" and count_k == 301:
                #         print("self.netDe.n_primitives: ", self.netDe.n_primitives)
                #         base_name = test_loader.dataset.label_to_category(label_point_cloud[k]) + "_" + str(
                #             count_k) + "_" + "complement" + "_pc" + ".ply"
                #         template_path = os.path.join(save_ply_path, base_name)
                #         save_ply(out_px1[k].cpu().detach().numpy(), path=template_path)
                #         for at_primitives in range(self.netDe.n_primitives):
                #             print("at_primitives: ", at_primitives)
                #             base_name = test_loader.dataset.label_to_category(label_point_cloud[k]) + "_" + str(
                #                 count_k) + "_" + str(at_primitives) + "_pc" + ".ply"
                #             template_path = os.path.join(save_ply_path, base_name)
                #             plane_points = self.netDe.atlas_primitives[at_primitives][0].transpose(1, 0)
                #             save_ply(plane_points.cpu().detach().numpy(), path=template_path)
                #         return 1
                if self.parameter["combined_pc"]:
                    pc_point_completion_dense = torch.cat([partial_point_cloud, out_px1], dim=1)
                    cd_loss_dense = self.l_cd(gt_point_cloud, pc_point_completion_dense)
                    # partial_point_cloud_sparse, _, _ = farthest_point_sampling(partial_point_cloud, ratio=0.5)
                    # out_px1_sparse, _, _ = farthest_point_sampling(out_px1, ratio=0.5)
                    # pc_point_completion_sparse = torch.cat([partial_point_cloud_sparse, out_px1_sparse], dim=1)
                    cd_loss_sparse = cd_loss_dense

                evaluate_loss_sparse.append(cd_loss_sparse.item())
                evaluate_loss_dense.append(cd_loss_dense.item())

                for k in range(partial_point_cloud.shape[0]):
                    class_name_choice = PointCompletionShapeNet.evaluation_class(
                        label_name=test_loader.dataset.label_to_category(label_point_cloud[k]))
                    evaluate_class_choice_sparse[class_name_choice].append(cd_loss_sparse.item())
                    evaluate_class_choice_dense[class_name_choice].append(cd_loss_dense.item())

                if self.parameter["gene_file"]:
                    for k in range(partial_point_cloud.shape[0]):
                        base_name = test_loader.dataset.label_to_category(label_point_cloud[k]) + "_" + str(
                            count_k) + "_pc_recon" + ".ply"
                        template_path = os.path.join(save_ply_path, base_name)
                        save_ply(pc=pc_point_completion_dense[k].cpu().detach().numpy(), path=template_path)

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

        # print(
        #     "====> cd_sparse: Airplane: {:.6f}, Cabinet: {:.6f}, Car: {:.6f}, Chair: {:.6f}, Lamp: {:.6f}, Sofa: {:.6f}, Table: {:.6f}, Watercraft: {:.6f}, mean: {:.6f}".format(
        #         evaluate_class_choice_sparse["Plane"], evaluate_class_choice_sparse["Cabinet"],
        #         evaluate_class_choice_sparse["Car"], evaluate_class_choice_sparse["Chair"],
        #         evaluate_class_choice_sparse["Lamp"], evaluate_class_choice_sparse["Couch"],
        #         evaluate_class_choice_sparse["Table"], evaluate_class_choice_sparse["Watercraft"],
        #         sum(evaluate_loss_sparse) / len(evaluate_loss_sparse)))
        # print(
        #     "====> cd_dense: Airplane: {:.6f}, Cabinet: {:.6f}, Car: {:.6f}, Chair: {:.6f}, Lamp: {:.6f}, Sofa: {:.6f}, Table: {:.6f}, Watercraft: {:.6f}, mean: {:.6f}".format(
        #         evaluate_class_choice_dense["Plane"], evaluate_class_choice_dense["Cabinet"],
        #         evaluate_class_choice_dense["Car"], evaluate_class_choice_dense["Chair"],
        #         evaluate_class_choice_dense["Lamp"], evaluate_class_choice_dense["Couch"],
        #         evaluate_class_choice_dense["Table"], evaluate_class_choice_dense["Watercraft"],
        #         sum(evaluate_loss_dense) / len(evaluate_loss_dense)))

        return sum(evaluate_loss_sparse) / len(evaluate_loss_sparse), sum(evaluate_loss_dense) / len(
            evaluate_loss_dense)

    def evaluation_step_kitti(self, test_loader, check_point_name=None):
        if self.parameter["gene_file"]:
            save_ply_path = os.path.join(os.path.dirname(__file__), "../../save_kitti_ply_data")
            make_dirs(save_ply_path)
            if check_point_name is not None:
                check_point_base_name = check_point_name.split(".")
                save_ply_path = os.path.join(os.path.dirname(__file__), "../../save_kitti_ply_data",
                                             check_point_base_name[0])
                make_dirs(save_ply_path)
        print("save_ply_path: ", save_ply_path)
        count_k = 0
        if count_k >= 200:
            return
        for i, dataset in enumerate(test_loader):
            gt_point_cloud, partial_point_cloud, label_point_cloud = dataset
            gt_point_cloud = gt_point_cloud.to(self.device)
            partial_point_cloud = partial_point_cloud.to(self.device)
            with torch.no_grad():
                self.eval()
                out_px3, out_px2, out_px1, _ = self(partial_point_cloud)
                pc_point_completion_dense = torch.cat([partial_point_cloud, out_px1], dim=1)
                if self.parameter["gene_file"]:
                    for k in range(partial_point_cloud.shape[0]):
                        base_name = "kitti_car" + "_" + str(count_k) + "_pc_recon" + ".ply"
                        template_path = os.path.join(save_ply_path, base_name)
                        save_ply(pc=pc_point_completion_dense[k].cpu().detach().numpy(), path=template_path)

                        base_name = "kitti_car" + "_" + str(count_k) + "_pc" + ".ply"
                        template_path = os.path.join(save_ply_path, base_name)
                        save_ply(partial_point_cloud[k].cpu().detach().numpy(), path=template_path)
                        count_k += 1
