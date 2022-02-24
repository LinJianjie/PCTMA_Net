from abc import ABC, abstractmethod
import time
from components.Logger import *
import torch.nn as nn
import torch.optim as optim
import torch
import os
from collections import OrderedDict


class BaseModel(nn.Module):
    def __init__(self, parameter, checkpoint_name, best_name, checkpoint_path, logger_file_name):
        super(BaseModel, self).__init__()
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

    def forward(self, *args):
        pass

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3, betas=(0.9, 0.999))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.parameter["epochs"],
                                                               eta_min=1e-3)
        opt_scheduler = OrderedDict({"opt": optimizer, "scheduler": scheduler})
        self.model_optimizer = opt_scheduler
        return opt_scheduler

    @abstractmethod
    def toCuda(self, device):
        pass

    def save_checkpoint(self, state, best_model_name):
        # time_str = time.strftime("%Y%m%d_%H%M%S")
        best_model = self.time_str + "_" + best_model_name
        print("--> best_model: ", best_model)
        self.best_name = best_model
        save_path = os.path.join(self.checkpoint_path, best_model)
        torch.save(state, save_path)

    @abstractmethod
    def load_checkpoint(self, filename="checkpoint"):
        pass

    @abstractmethod
    def train_one_epoch(self, train_loader, at_epoch, n_epochs, batch_size):
        pass

    @abstractmethod
    def train_step(self, start_epoch, n_epochs, train_loader, test_loader, best_loss=0.0, batch_size=8,
                   best_model_name="best_model.pth"):
        pass

    @abstractmethod
    def evaluation_step(self, test_loader, check_point_name=None):
        pass

    @abstractmethod
    def backward_model(self, *args):
        pass

    @abstractmethod
    def backward_model_update(self, loss):
        pass

    @staticmethod
    def configure_loss_function():
        pass

    def count_parameters(self):
        number_of_parameter = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("the number of parameter:{:.2f} ".format(number_of_parameter / 1e6))
