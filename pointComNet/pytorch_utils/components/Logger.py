import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
import logging


class Logger:
    def __init__(self, logname="log", level="INFO", use_console=True, use_file=True, filename="log.log"):
        self.formatter = logging.Formatter(
            '%(asctime)s - %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
        if level == "INFO":
            self.level = logging.INFO
        if level == "DEBUG":
            self.level = logging.DEBUG
        if level == "WARNING":
            self.level = logging.warning

        self.logger = logging.getLogger(logname)
        self.logger.setLevel(logging.INFO)
        if use_console:
            self.__add_console()

        if use_file:
            # self.filename = "log/"+filename
            self.filename = filename
            self.__add_file()

    def __add_console(self):
        console = logging.StreamHandler()
        console.setLevel(self.level)
        console.setFormatter(self.formatter)
        self.logger.addHandler(console)

    def __add_file(self):
        file = logging.FileHandler(filename=self.filename, mode='w')
        file.setLevel(self.level)
        file.setFormatter(self.formatter)
        self.logger.addHandler(file)

    def INFO(self, msg, *args, **kwargs):
        self.logger.info(msg, *args, **kwargs)
