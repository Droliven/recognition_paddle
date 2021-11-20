#!/usr/bin/env python
# encoding: utf-8
'''
@project : PaddleVideo
@file    : skeleton.py
@author  : Levon
@contact : levondang@163.com
@ide     : PyCharm
@time    : 2021-11-18 00:41
'''

import os.path as osp
import copy
import random
import numpy as np
import pickle
import math

import paddle
from paddle.io import Dataset, DataLoader


class SkeletonDataset(Dataset):

    def __init__(self, file_path, label_path=None, mode="train", segment=100, repeat=20):
        super(Dataset, self).__init__()

        self.file_path = file_path # [2922/628, 3, 2500, 25, 1]
        self.label_path = label_path
        self.segment = segment
        self.mode = mode
        self.repeat = repeat

        """Load feature file to get skeleton information."""
        print("Loading data, it will take some moment...")
        self.data = np.load(self.file_path)
        if mode == "train" and label_path is not None:
            self.label = np.load(self.label_path)

        print("Data Loaded!")

        self.confidence = self.data[:, 2:, :, :, :]  # n, 1, 2500, 25, 1
        # 相对化，二维化
        self.data = self.data[:, :2, :, :, :]
        self.data = self.data - self.data[:, :, :, 8:9, :]


    def get_sample_len(self):
        # train: [26, 2037]; test: [42, 1716]
        min_T = 10000
        max_T = 0
        for i in range(self.data.__len__()):
            d = self.data[i]  # 2, 2500, 25, 1
            T = self.get_valid_t(d)
            if T > max_T:
                max_T = T
            if T < min_T:
                min_T = T
            print("idx {:>5d}, len {:>8d}".format(i, T))  # T是有效帧的后一帧
        print(f"{min_T}, {max_T}")

    def get_valid_t(self, d):
        C, T, V, M = d.shape
        for i in range(T - 1, -1, -1):
            tmp = np.sum(d[:, i, :, :])
            if tmp > 0:
                T = i + 1
                break
        return T


    def get_min_max(self):
        '''
        train:[-0.7258526682853699, 0.5368971824645996], [-0.9634618163108826, 0.8484165072441101], label: [0, 29]
        test: [-0.5388184189796448, 0.5079047679901123], [-0.9860285520553589, 0.7925562858581543]
        Returns:

        '''
        minl, maxl = np.min(self.label), np.max(self.label)
        print(f"[{minl}, {maxl}]")  # train:

        minc, maxc = np.min(self.confidence), np.max(self.confidence)
        print(f"[{minc}, {maxc}]")  # train:

        x_period = [1000, -1000]
        y_period = [1000, -1000]
        for i in range(self.data.__len__()):
            d = self.data[i]  # 2, 2500, 25, 1
            C, T, V, M = d.shape
            minx, maxx = np.min(d[0]), np.max(d[0])
            miny, maxy = np.min(d[1]), np.max(d[1])
            if minx < x_period[0]:
                x_period[0] = minx
            if maxx > x_period[1]:
                x_period[1] = maxx
            if miny < y_period[0]:
                y_period[0] = miny
            if maxy > y_period[1]:
                y_period[1] = maxy

        print(f"[{x_period[0]}, {x_period[1]}], [{y_period[0]}, {y_period[1]}]")  # train: 2037; test: 1716

    def __len__(self):
        """get the size of the dataset."""
        return len(self.data)

    def get_train(self, idx):
        """ Get the sample for either training or testing given index"""
        data = copy.deepcopy(self.data[idx])
        confidence = copy.deepcopy(self.confidence[idx])

        dl = self.get_valid_t(data)

        # segment
        if dl <= self.segment:
            data = data[:, :self.segment, :, :]
            confidence = confidence[:, :self.segment, :, :]
        else:
            base_num = dl // self.segment
            some_num = dl - (base_num * self.segment)
            seg_idx = []
            start = 0
            for t in range(some_num):
                randidx = random.randint(0, base_num)
                seg_idx.append(start + randidx)
                start += (base_num + 1)
            for t in range(some_num, self.segment):
                randidx = random.randint(0, base_num - 1)
                seg_idx.append(start + randidx)
                start += base_num
            data = data[:, seg_idx, :, :]
            confidence = confidence[:, seg_idx, :, :]

        label = copy.deepcopy(self.label[idx])
        return [data, confidence, label]

    def get_test(self, idx):
        """ Get the sample for either training or testing given index"""
        data = copy.deepcopy(self.data[idx]) # 2, 2500, 25, 1
        confidence = copy.deepcopy(self.confidence[idx])
        dl = self.get_valid_t(data)

        repeat_data = []
        repeat_confidence = []
        for h in range(self.repeat):
            # segment
            if dl <= self.segment:
                choose_data = data[:, :self.segment, :, :]
                choose_confidence = confidence[:, :self.segment, :, :]
            else:
                base_num = dl // self.segment
                some_num = dl - (base_num * self.segment)
                seg_idx = []
                start = 0
                for t in range(some_num):
                    randidx = random.randint(0, base_num)
                    seg_idx.append(start + randidx)
                    start += (base_num + 1)
                for t in range(some_num, self.segment):
                    randidx = random.randint(0, base_num - 1)
                    seg_idx.append(start + randidx)
                    start += base_num
                choose_data = data[:, seg_idx, :, :]
                choose_confidence = confidence[:, seg_idx, :, :]

            repeat_data.append(choose_data)
            repeat_confidence.append(choose_confidence)

        repeat_data = np.stack(repeat_data, axis=0)  # 10, 2, 100, 25, 1
        repeat_confidence = np.stack(repeat_confidence, axis=0)  # 10, 2, 100, 25, 1

        return [repeat_data, repeat_confidence]


    def __getitem__(self, idx):
        if self.mode == "train":
            result = self.get_train(idx)
        elif self.mode == "test":
            result = self.get_test(idx)
        return result


if __name__ == '__main__':
    from levon_recognition.agcn.datas.draw_pictures import draw_pic_single_2d

    I25 = [8, 12, 13, 14, 19, 14, 8, 9, 10, 11, 22, 11, 8, 1, 0, 15, 0, 16, 1, 2, 3, 1, 5, 6]
    J25 = [12, 13, 14, 19, 20, 21, 9, 10, 11, 22, 23, 24, 1, 0, 15, 17, 16, 18, 2, 3, 4, 5, 6, 7]
    LR25 = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0]

    # ds = SkeletonDataset(file_path=r'E:\\second_model_report_data\\datas\\paddle_action_recognization\\train_dataset\\train_data.npy', label_path=r'E:\\second_model_report_data\\datas\\paddle_action_recognization\\train_dataset\\train_label.npy', mode="train")
    ds = SkeletonDataset(file_path=r'E:\\second_model_report_data\\datas\\paddle_action_recognization\\test_A_data\\test_A_data.npy', label_path=None, mode="test")

    # ds.get_sample_len()
    # ds.get_min_max()

    dl = DataLoader(dataset=ds, batch_size=1, shuffle=False)

    for i, data in enumerate(dl): # b, [repeats], 2, 100, 25, 1
        d2draw = paddle.transpose(data[0, 0, :, 0, :, 0], [1, 0]).numpy()
        draw_pic_single_2d(d2draw, I=I25, J=J25, LR=LR25, full_path=f"{i}.png")
        pass
    pass