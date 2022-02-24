import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from abc import ABC, abstractmethod
from components.ioUtils import *


class MainAPP(object):
    def __init__(self):
        self.log_dir = None
        self.check_point_dir = None
        self.model = None
        self.train_model = None

    @abstractmethod
    def load_data(self, **kwargs):
        pass

    @abstractmethod
    def read_parameters(self, **kwargs):
        pass

    @abstractmethod
    def define_model(self, **kwargs):
        pass

    @abstractmethod
    def define_train_model(self, **kwargs):
        pass

    @abstractmethod
    def load_check_points(self):
        pass

    @abstractmethod
    def create_log(self):
        self.log_dir = make_dirs_log()

    @abstractmethod
    def create_check_points(self):
        self.check_point_dir = make_dirs_checkout()

    @abstractmethod
    def train(self, **kwargs):
        pass

    @abstractmethod
    def eval(self, **kwargs):
        pass
