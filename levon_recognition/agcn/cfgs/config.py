#!/usr/bin/env python
# encoding: utf-8
'''
@project : PaddleVideo
@file    : config.py
@author  : Levon
@contact : levondang@163.com
@ide     : PyCharm
@time    : 2021-11-18 20:37
'''
from getpass import getuser
import os


class Config():
    def __init__(self):
        self.platform = getuser()
        self.exp_name = "agcn"

        # data
        if self.platform == "aistudio":
            self.train_data_path = os.path.join("/home/aistudio/data", "data104925", "train_data.npy")
            self.train_label_path = os.path.join("/home/aistudio/data", "data104925", "train_label.npy")
            self.test_data_path = os.path.join("/home/aistudio/data", "data104924", "test_A_data.npy")
        elif self.platform == "Drolab":
            self.train_data_path = os.path.join(r"E:\second_model_report_data\datas\paddle_action_recognization", "train_dataset", "train_data.npy")
            self.train_label_path = os.path.join(r"E:\second_model_report_data\datas\paddle_action_recognization", "train_dataset", "train_label.npy")
            self.test_data_path = os.path.join(r"E:\second_model_report_data\datas\paddle_action_recognization", "test_A_data", "test_A_data.npy")

        self.I25 = [ 8, 12, 13, 14, 19, 14, 8,  9, 10, 11, 22, 11, 8, 1,  0, 15,  0, 16, 1, 2, 3, 1, 5, 6]
        self.J25 = [12, 13, 14, 19, 20, 21, 9, 10, 11, 22, 23, 24, 1, 0, 15, 17, 16, 18, 2, 3, 4, 5, 6, 7]
        self.LR25 = [0,  0,  0,  0,  0,  0, 1,  1,  1,  1,  1,  1, 0, 0,  1,  1,  0,  0, 1, 1, 1, 0, 0, 0]

        self.segment = 300
        self.repeat = 15

        self.train_batch_size = 32
        self.test_batch_size = 1

        self.num_workers = 4
        self.lr = 1e-3
        self.n_epoch = 150

        if self.platform == "aistudio":
            self.ckpt_base_dir = os.path.join("/home/aistudio/work/outputs", "ckpt")
        elif self.platform == 'Drolab':
            self.ckpt_base_dir = os.path.join(r"E:\PythonWorkspace\PaddleVideo\levon_recognition", "ckpt")

        if not os.path.exists(self.ckpt_base_dir):
            os.makedirs(self.ckpt_base_dir)

        if not os.path.exists(os.path.join(self.ckpt_base_dir, "images")):
            os.makedirs(os.path.join(self.ckpt_base_dir, "images"))

        if not os.path.exists(os.path.join(self.ckpt_base_dir, "models")):
            os.makedirs(os.path.join(self.ckpt_base_dir, "models"))


if __name__ == '__main__':
    u = getuser()
    pass