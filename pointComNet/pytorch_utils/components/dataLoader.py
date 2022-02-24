import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
import glob
from components.constant import *


def Completion3Ddownload():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, dataFile, "dataset2019")
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(os.path.join(DATA_DIR, 'shapenet')):
        www = 'http://download.cs.stanford.edu/downloads/completion3d/dataset2019.zip'
        zipfile = os.path.basename(www)
        os.system('wget --no-check-certificate %s; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % ("shapenet", DATA_DIR))
        os.system('rm %s' % zipfile)


def Completion3Ddownload16K():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, dataFile)
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(os.path.join(DATA_DIR, 'shapenet')):
        www = 'http:// download.cs.stanford.edu/downloads/completion3d/shapenet16K2019.zip'
        zipfile = os.path.basename(www)
        os.system('wget --no-check-certificate %s; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
        os.system('rm %s' % zipfile)


def ModelNet40download():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, dataFile, 'ModelNet40')
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
        www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
        zipfile = os.path.basename(www)
        os.system('wget --no-check-certificate %s; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
        os.system('rm %s' % zipfile)


def KITTIDownload():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, dataFile, 'Kitti')
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
        www = "https://drive.google.com/drive/folders/1fSu0_huWhticAlzLh3Ejpg8zxzqO1z-F?usp=sharing"


def ShapeNetDownload():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, dataFile, 'ShapeNetPart')
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(os.path.join(DATA_DIR, 'ShapeNetPart')):
        url = "https://shapenet.cs.stanford.edu/media/shapenet_part_seg_hdf5_data.zip"
        zipfile = os.path.basename(url)
        shapeNetName = "hdf5_data"
        os.system('wget --no-check-certificate %s; unzip %s' % (url, zipfile))
        os.system('mv %s %s' % (shapeNetName, "ShapeNetPart"))
        os.system('rm %s' % zipfile)


def YCBDatasetDownload():
    pass
