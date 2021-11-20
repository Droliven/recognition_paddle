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
    def __init__(self, exp_name="msmrgcn"):
        self.exp_name = exp_name
        self.platform = getuser()

        # data
        if self.platform == "aistudio":
            # self.train_data_path = os.path.join("/home/aistudio/data", "data104925", "train_data.npy")
            # self.train_label_path = os.path.join("/home/aistudio/data", "data104925", "train_label.npy")
            # self.test_data_path = os.path.join("/home/aistudio/data", "data104924", "test_A_data.npy")

            self.train_data_path = os.path.join(
                r"/home/aistudio/data/new_datas", "new_train_data.npy")
            self.train_label_path = os.path.join(
                r"/home/aistudio/data/new_datas", "new_train_label.npy")
            self.val_data_path = os.path.join(
                r"/home/aistudio/data/new_datas", "new_val_data.npy")
            self.val_label_path = os.path.join(
                r"/home/aistudio/data/new_datas", "new_val_label.npy")
            self.test_A_data_path = os.path.join(
                r"/home/aistudio/data/new_datas", "test_A_data.npy")

        elif self.platform == "Drolab":
            self.train_data_path = os.path.join(r"E:\second_model_report_data\datas\paddle_action_recognization\new_datas", "new_train_data.npy")
            self.train_label_path = os.path.join(r"E:\second_model_report_data\datas\paddle_action_recognization\new_datas", "new_train_label.npy")
            self.val_data_path = os.path.join(r"E:\second_model_report_data\datas\paddle_action_recognization\new_datas", "new_val_data.npy")
            self.val_label_path = os.path.join(r"E:\second_model_report_data\datas\paddle_action_recognization\new_datas", "new_val_label.npy")
            self.test_A_data_path = os.path.join(r"E:\second_model_report_data\datas\paddle_action_recognization\new_datas", "test_A_data.npy")

        elif self.platform == "songbo":
            # self.train_data_path = os.path.join(
            #     r"/home/ml_group/songbo/danglingwei/new_datas", "new_train_data.npy")
            # self.train_label_path = os.path.join(
            #     r"/home/ml_group/songbo/danglingwei/new_datas", "new_train_label.npy")
            # self.val_data_path = os.path.join(
            #     r"/home/ml_group/songbo/danglingwei/new_datas", "new_val_data.npy")
            # self.val_label_path = os.path.join(
            #     r"/home/ml_group/songbo/danglingwei/new_datas", "new_val_label.npy")
            # self.test_A_data_path = os.path.join(
            #     r"/home/ml_group/songbo/danglingwei/new_datas", "test_A_data.npy")

            self.train_data_path = os.path.join(
                r"/home/songbo/danglingwei/new_datas", "new_train_data.npy")
            self.train_label_path = os.path.join(
                r"/home/songbo/danglingwei/new_datas", "new_train_label.npy")
            self.val_data_path = os.path.join(
                r"/home/songbo/danglingwei/new_datas", "new_val_data.npy")
            self.val_label_path = os.path.join(
                r"/home/songbo/danglingwei/new_datas", "new_val_label.npy")
            self.test_A_data_path = os.path.join(
                r"/home/songbo/danglingwei/new_datas", "test_A_data.npy")


        self.I25 = [ 8, 12, 13, 14, 19, 14, 8,  9, 10, 11, 22, 11, 8, 1,  0, 15,  0, 16, 1, 2, 3, 1, 5, 6]
        self.J25 = [12, 13, 14, 19, 20, 21, 9, 10, 11, 22, 23, 24, 1, 0, 15, 17, 16, 18, 2, 3, 4, 5, 6, 7]
        self.LR25 = [0,  0,  0,  0,  0,  0, 1,  1,  1,  1,  1,  1, 0, 0,  1,  1,  0,  0, 1, 1, 1, 0, 0, 0]

        self.down25_11 = [[0, 15, 16, 17, 18], [1], [2, 3], [3, 4], [5, 6], [6, 7], [8, 9, 12], [10], [11, 22, 23, 24], [13], [14, 19, 20, 21]]
        self.down11_5 = [[0, 1], [2, 3], [4, 5], [6, 7, 8], [6, 9, 10]]

        self.segment = 300
        self.repeat = 15

        # self.val_mode = "val_single"
        self.val_mode = "val_repeat"
        self.train_batch_size = 16
        self.test_batch_size = 1
        self.val_batch_size = 4

        self.multi_scale_node_n = [25, 11, 5]
        self.multi_scale_hidden_dim = [400, 200, 100]

        self.num_workers = 2
        self.lr = 1e-3
        self.n_epoch = 150
        self.dropout_rate = 0.2

        if self.platform == "aistudio":
            self.ckpt_base_dir = os.path.join("/home/aistudio/work/outputs", exp_name)
        elif self.platform == 'Drolab':
            self.ckpt_base_dir = os.path.join(r"E:\PythonWorkspace\recognition_paddle", "ckpt", exp_name)
        elif self.platform == "songbo":
            # self.ckpt_base_dir = os.path.join(r"/home/ml_group/songbo/danglingwei", "outputs", exp_name)
            self.ckpt_base_dir = os.path.join(r"/home/songbo/danglingwei", "outputs", exp_name)


        if not os.path.exists(self.ckpt_base_dir):
            os.makedirs(self.ckpt_base_dir)

        if not os.path.exists(os.path.join(self.ckpt_base_dir, "images")):
            os.makedirs(os.path.join(self.ckpt_base_dir, "images"))

        if not os.path.exists(os.path.join(self.ckpt_base_dir, "models")):
            os.makedirs(os.path.join(self.ckpt_base_dir, "models"))

class ConfigDCT():
    def __init__(self, exp_name="msmrgcn"):
        self.exp_name = exp_name
        self.platform = getuser()

        # data
        if self.platform == "aistudio":
            # self.train_data_path = os.path.join("/home/aistudio/data", "data104925", "train_data.npy")
            # self.train_label_path = os.path.join("/home/aistudio/data", "data104925", "train_label.npy")
            # self.test_data_path = os.path.join("/home/aistudio/data", "data104924", "test_A_data.npy")

            self.train_data_path = os.path.join(
                r"/home/aistudio/data/new_datas", "new_train_data.npy")
            self.train_label_path = os.path.join(
                r"/home/aistudio/data/new_datas", "new_train_label.npy")
            self.val_data_path = os.path.join(
                r"/home/aistudio/data/new_datas", "new_val_data.npy")
            self.val_label_path = os.path.join(
                r"/home/aistudio/data/new_datas", "new_val_label.npy")
            self.test_A_data_path = os.path.join(
                r"/home/aistudio/data/new_datas", "test_A_data.npy")

        elif self.platform == "Drolab":
            self.train_data_path = os.path.join(r"E:\second_model_report_data\datas\paddle_action_recognization\new_datas", "new_train_data.npy")
            self.train_label_path = os.path.join(r"E:\second_model_report_data\datas\paddle_action_recognization\new_datas", "new_train_label.npy")
            self.val_data_path = os.path.join(r"E:\second_model_report_data\datas\paddle_action_recognization\new_datas", "new_val_data.npy")
            self.val_label_path = os.path.join(r"E:\second_model_report_data\datas\paddle_action_recognization\new_datas", "new_val_label.npy")
            self.test_A_data_path = os.path.join(r"E:\second_model_report_data\datas\paddle_action_recognization\new_datas", "test_A_data.npy")

        elif self.platform == "songbo":
            # self.train_data_path = os.path.join(
            #     r"/home/ml_group/songbo/danglingwei/new_datas", "new_train_data.npy")
            # self.train_label_path = os.path.join(
            #     r"/home/ml_group/songbo/danglingwei/new_datas", "new_train_label.npy")
            # self.val_data_path = os.path.join(
            #     r"/home/ml_group/songbo/danglingwei/new_datas", "new_val_data.npy")
            # self.val_label_path = os.path.join(
            #     r"/home/ml_group/songbo/danglingwei/new_datas", "new_val_label.npy")
            # self.test_A_data_path = os.path.join(
            #     r"/home/ml_group/songbo/danglingwei/new_datas", "test_A_data.npy")

            self.train_data_path = os.path.join(
                r"/home/songbo/danglingwei/new_datas", "new_train_data.npy")
            self.train_label_path = os.path.join(
                r"/home/songbo/danglingwei/new_datas", "new_train_label.npy")
            self.val_data_path = os.path.join(
                r"/home/songbo/danglingwei/new_datas", "new_val_data.npy")
            self.val_label_path = os.path.join(
                r"/home/songbo/danglingwei/new_datas", "new_val_label.npy")
            self.test_A_data_path = os.path.join(
                r"/home/songbo/danglingwei/new_datas", "test_A_data.npy")


        self.I25 = [ 8, 12, 13, 14, 19, 14, 8,  9, 10, 11, 22, 11, 8, 1,  0, 15,  0, 16, 1, 2, 3, 1, 5, 6]
        self.J25 = [12, 13, 14, 19, 20, 21, 9, 10, 11, 22, 23, 24, 1, 0, 15, 17, 16, 18, 2, 3, 4, 5, 6, 7]
        self.LR25 = [0,  0,  0,  0,  0,  0, 1,  1,  1,  1,  1,  1, 0, 0,  1,  1,  0,  0, 1, 1, 1, 0, 0, 0]

        self.down25_11 = [[0, 15, 16, 17, 18], [1], [2, 3], [3, 4], [5, 6], [6, 7], [8, 9, 12], [10], [11, 22, 23, 24], [13], [14, 19, 20, 21]]
        self.down11_5 = [[0, 1], [2, 3], [4, 5], [6, 7, 8], [6, 9, 10]]

        self.dct_n = 30

        self.train_batch_size = 32
        self.test_batch_size = 1
        self.val_batch_size = 32

        self.multi_scale_node_n = [25, 11, 5]
        self.multi_scale_hidden_dim = [64, 128, 256]

        self.num_workers = 8
        self.lr = 1e-3
        self.n_epoch = 150
        self.dropout_rate = 0.2

        if self.platform == "aistudio":
            self.ckpt_base_dir = os.path.join("/home/aistudio/work/outputs", "ckpt", exp_name)
        elif self.platform == 'Drolab':
            self.ckpt_base_dir = os.path.join(r"E:\PythonWorkspace\recognition_paddle", "ckpt", exp_name)
        elif self.platform == "songbo":
            # self.ckpt_base_dir = os.path.join(r"/home/ml_group/songbo/danglingwei", "outputs", exp_name)
            self.ckpt_base_dir = os.path.join(r"/home/songbo/danglingwei", "outputs", "ckpt", exp_name)


        if not os.path.exists(self.ckpt_base_dir):
            os.makedirs(self.ckpt_base_dir)

        if not os.path.exists(os.path.join(self.ckpt_base_dir, "images")):
            os.makedirs(os.path.join(self.ckpt_base_dir, "images"))

        if not os.path.exists(os.path.join(self.ckpt_base_dir, "models")):
            os.makedirs(os.path.join(self.ckpt_base_dir, "models"))

if __name__ == '__main__':
    u = getuser()
    pass