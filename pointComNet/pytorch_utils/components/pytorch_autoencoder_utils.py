import os
import shutil
import logging
import torch
import numpy as np
import torch.nn.functional as F
import sklearn.metrics as metrics
from components.Logger import Logger
from components.ioUtils import load_checkpoint, save_ply


class AutoEncoderTrain:
    def __int__(self, model, loss_function=None, optimizer=None, checkpoint_name="ckpt", best_name="best",
                lr_scheduler=None,
                eval_frequency=-1,
                loggerfilename="log.log"):
        self.model = model
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.checkpoint_name = checkpoint_name
        self.best_name = best_name
        self.lr_scheduler = lr_scheduler
        self.global_epoch = 0
        self.global_step = 0
        self.best_instance_acc = 100000
        self.best_class_acc = 100000
        self.mean_correct = []
        self.best_epoch = 0
        self.n_epochs = 0
        self.Logger = Logger(filename=loggerfilename)
        self.save_path = None

    def train_single_epoch(self, train_loader, device, at_epoach, n_eposchs, batch_size):
        train_correct = 0
        count = 0.0
        train_loss = 0
        train_pred = []
        train_true = []
        self.model.train()
        for i, dataset in enumerate(train_loader):
            batch_data, label = dataset
            batch_data = batch_data.to(device)
            label = label.to(device)
            self.optimizer.zero_grad()
            self.model.train()
            model_output = self.model(batch_data)
            loss = self.model.loss(batch_data, model_output, self.loss_function)
            train_loss += loss.item() * batch_size
            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()

    def eval_epoch(self, data_loader, device, epoch, batch_size, best_model_name="best_model.pth"):
        self.model.eval()
        correct = 0
        test_pred = []
        test_true = []
        count = 0.0
        test_loss = 0
        batch_size = 0
        for i, dataset in enumerate(data_loader):
            batch_data, label = dataset
            batch_data = batch_data.to(device)
            label = label.to(device)
            with torch.no_grad():
                model_output = self.model(batch_data)
                loss = self.model.loss(batch_data, model_output, self.loss_function)
                test_loss += loss.item() * batch_size
        if test_loss < self.best_instance_acc:
            self.best_instance_acc = test_loss
            self.best_epoch = epoch
            checkpointPath = os.path.join(os.getcwd(), '../checkpoints')
            if not os.path.exists(checkpointPath):
                os.mkdir(checkpointPath)
            self.save_path = os.path.join(checkpointPath, best_model_name)
            state = {
                "batch_size": batch_size,
                "epoch": self.best_epoch,
                "instance_acc": self.best_instance_acc,
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict()}
            torch.save(state, self.save_path)

    def test(self, test_loader, device, to_save_ply):
        load_checkpoint(self.model, self.optimizer, self.save_path)
        correct = 0
        test_pred = []
        test_true = []
        count = 0.0
        test_loss = 0
        batch_size = 0
        autoEncoderPath = os.path.join(os.getcwd(), '../AutoEncoder_results')
        if not os.path.exists(autoEncoderPath):
            os.mkdir(autoEncoderPath)
        for i, dataset in enumerate(test_loader):
            batch_data, labels = dataset
            batch_data = batch_data.to(device)
            with torch.no_grad():
                model_output = self.model(batch_data)
                if to_save_ply:
                    for k, label in enumerate(labels):
                        for key, value in test_loader.dataset.shape_name:
                            if i == value:
                                class_name_ = key
                                real_path = os.path.join(autoEncoderPath, 'orig_{}_{}.ply'.format(class_name_, k))
                                save_ply(real_path, batch_data[k])
                                genera_path = os.path.join(autoEncoderPath, 'gen_{}_{}.ply'.format(class_name_, k))
                                save_ply(genera_path, model_output[k])
                                break

                loss = self.model.loss(batch_data, model_output, self.loss_function)

    def train(self, start_epoch, n_epochs, device, train_loader, test_loader=None, best_loss=0.0, batch_size=8,
              best_model_name="best_model.pth"):
        self.n_epochs = n_epochs
        for at_epoch in range(start_epoch, n_epochs):
            self.train_single_epoch(train_loader=train_loader, device=device,
                                    at_epoch=at_epoch,
                                    n_epochs=n_epochs,
                                    batch_size=batch_size)
            self.eval_epoch(test_loader, device, at_epoch, best_model_name)
