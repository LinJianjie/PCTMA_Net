import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from components.pointNetmodel.pointnet_util import *


class PointNetPPClassification(nn.Module):
    def __init__(self, num_class, config):
        super(PointNetPPClassification).__init__()
        self.num_class = num_class
        self.config = config
        self.sa1 = PointNetSetAbstractionMsg(ratio=self.config["sa1"]["ratio"],
                                             radius_list=self.config["sa1"]["radius_list"],
                                             max_sample_list=self.config["sa1"]["max_sample_list"],
                                             mlp_list=self.config["sa1"]["mlp_list"])

        self.sa2 = PointNetSetAbstractionMsg(ratio=self.config["sa2"]["ratio"],
                                             radius_list=self.config["sa2"]["radius_list"],
                                             max_sample_list=self.config["sa2"]["max_sample_list"],
                                             mlp_list=self.config["sa2"]["mlp_list"])

        self.sa3 = PointNetSetAbstraction(ratio=self.config["sa3"]["ratio"],
                                          radius=self.config["sa3"]["radius"],
                                          nsample=self.config["sa3"]["nsample"],
                                          mlp_list=self.config["sa3"]["mlp_list"],
                                          group_all=self.config["sa3"]["group_all"])
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, self.num_class)

    def forward(self, xyz, features):
        B, _, _ = xyz.shape
        l1_xyz, l1_points = self.sa1(xyz, features)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        x = F.log_softmax(x, -1)
        return x, l3_points
