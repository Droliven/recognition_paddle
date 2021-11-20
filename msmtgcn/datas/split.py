#!/usr/bin/env python
# encoding: utf-8
'''
@project : t1111111111
@file    : split.py
@author  : Levon
@contact : levondang@163.com
@ide     : PyCharm
@time    : 2021-11-20 12:35
'''

import numpy as np
import os
import getpass

def split():
    if getpass.getuser() == "Drolab":
        train_data_path = r"E:\\second_model_report_data\\datas\\paddle_action_recognization\\train_dataset\\train_data.npy"
        train_label_path = r"E:\\second_model_report_data\\datas\\paddle_action_recognization\\train_dataset\\train_label.npy"
        out_dir = os.path.join(r"E:\second_model_report_data\datas\paddle_action_recognization", "new_datas")
    elif getpass.getuser() == "aistudio":
        train_data_path = os.path.join("/home/aistudio/data", "data104925", "train_data.npy")
        train_label_path = os.path.join("/home/aistudio/data", "data104925", "train_label.npy")
        out_dir = os.path.join("/home/aistudio/data", "new_datas")

    data = np.load(train_data_path)
    label = np.load(train_label_path)

    dl = data.shape[0]

    train_l = int(0.88 * dl)
    val_l = dl - train_l

    idx = np.arange(dl)
    np.random.seed(666666)
    np.random.shuffle(idx)

    new_train_data = data[idx[:train_l]]
    new_train_label = label[idx[:train_l]]

    new_val_data = data[idx[train_l:]]
    new_val_label = label[idx[train_l:]]

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    np.save(os.path.join(out_dir, "new_train_data.npy"), new_train_data)
    np.save(os.path.join(out_dir, "new_train_label.npy"), new_train_label)
    np.save(os.path.join(out_dir, "new_val_data.npy"), new_val_data)
    np.save(os.path.join(out_dir, "new_val_label.npy"), new_val_label)

def test():
    if getpass.getuser() == "Drolab":
        base_dir = r"E:\second_model_report_data\datas\paddle_action_recognization\new_datas"
        train_data_path = os.path.join(base_dir, "new_train_data.npy")
        train_label_path = os.path.join(base_dir, "new_train_label.npy")
        val_data_path = os.path.join(base_dir, "new_val_data.npy")
        val_label_path = os.path.join(base_dir, "new_val_label.npy")
        # test_A_data_path = os.path.join(base_dir, "test_A_data.npy")
    elif getpass.getuser() == "aistudio":
        base_dir = r"/home/aistudio/data/new_datas"
        train_data_path = os.path.join(base_dir, "new_train_data.npy")
        train_label_path = os.path.join(base_dir, "new_train_label.npy")
        val_data_path = os.path.join(base_dir, "new_val_data.npy")
        val_label_path = os.path.join(base_dir, "new_val_label.npy")
        # test_A_data_path = os.path.join(base_dir, "test_A_data.npy")

    train_data = np.load(train_data_path)
    train_label = np.load(train_label_path)
    val_data = np.load(val_data_path)
    val_label = np.load(val_label_path)
    # test_A_data = np.load(test_A_data_path)
    print(f"train_data: {train_data.shape}")
    print(f"train_label: {train_label.shape}")
    print(f"val_data: {val_data.shape}")
    print(f"val_label: {val_label.shape}")
    # print(f"test_A_data: {test_A_data.shape}")


split()
test()