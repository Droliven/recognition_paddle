#!/usr/bin/env python
# encoding: utf-8
'''
@project : t1111111111
@file    : msgcn.py
@author  : Levon
@contact : levondang@163.com
@ide     : PyCharm
@time    : 2021-11-19 22:06
'''

import paddle
import paddle.nn as nn
from collections import OrderedDict

from msgcn.nets.layers import GraphConv, GraphConvBlock, ResGCB


class MSGCN(nn.Layer):
    def __init__(self, in_len=300, multi_scale_node_n=[25, 11, 5], multi_scale_hidden_dim=[400, 200, 100], dropout_rate=0.1, down25_11 = [[0, 15, 16, 17, 18], [1], [2, 3], [3, 4], [5, 6], [6, 7], [8, 9, 12], [10], [11, 22, 23, 24], [13], [14, 19, 20, 21]], down11_5 = [[0, 1], [2, 3], [4, 5], [6, 7, 8], [6, 9, 10]]):
        super(MSGCN, self).__init__()

        self.multi_scale_node_n = multi_scale_node_n
        self.multi_scale_hidden_dim = multi_scale_hidden_dim

        self.down25_11_matrix = self.get_down_matrix(down_sample_idx=down25_11, from_n=25, to_n=11)
        self.down11_5_matrix = self.get_down_matrix(down_sample_idx=down11_5, from_n=11, to_n=5)
        self.up11_25_matrix = self.get_up_matrix(down_sample_idx=down25_11, from_n=11, to_n=25)
        self.up5_11_matrix = self.get_up_matrix(down_sample_idx=down11_5, from_n=5, to_n=11)

        self.pose2feature = GraphConvBlock(in_len=in_len, out_len=multi_scale_hidden_dim[0], in_node_n=multi_scale_node_n[0]*2, out_node_n=multi_scale_node_n[0]*2, dropout_rate=dropout_rate, bias=True, residual=False)

        unet = nn.LayerDict()
        recognitions = nn.LayerList()
        for i, (n, h) in enumerate(zip(multi_scale_node_n, multi_scale_hidden_dim)):
            n = n * 2
            unet.update({
                f"enc_s{i+1}": nn.Sequential(
                    # ResGCB(in_len=h, out_len=h, in_node_n=n, out_node_n=n, dropout_rate=dropout_rate, bias=True, residual=True),
                    ResGCB(in_len=h, out_len=h//2, in_node_n=n, out_node_n=n, dropout_rate=dropout_rate, bias=True, residual=False),
                    ResGCB(in_len=h//2, out_len=h//2, in_node_n=n, out_node_n=n, dropout_rate=dropout_rate, bias=True, residual=True),
                ),
                f"dec_s{i + 1}": nn.Sequential(
                    ResGCB(in_len=h//2, out_len=h//2, in_node_n=n, out_node_n=n, dropout_rate=dropout_rate, bias=True, residual=True),
                    ResGCB(in_len=h//2, out_len=h, in_node_n=n, out_node_n=n, dropout_rate=dropout_rate, bias=True, residual=False),
                    # ResGCB(in_len=h, out_len=h, in_node_n=n, out_node_n=n, dropout_rate=dropout_rate, bias=True, residual=True),
                ),
            })
            if i != 2:
                unet.update({
                    f"fuse_s{i + 1}": GraphConvBlock(in_len=h, out_len=h//2, in_node_n=n, out_node_n=n,
                                                     dropout_rate=dropout_rate, bias=True, residual=False),
                })

            recognitions.append(nn.Sequential(
                nn.Linear(n * h, h // 4),
                nn.GELU(),
                nn.Linear(h // 4, 30),
            ))
        self.unet = unet
        self.recognitions = recognitions


    def get_down_matrix(self, down_sample_idx=[[0, 15, 16, 17, 18], [1], [2, 3], [3, 4], [5, 6], [6, 7], [8, 9, 12], [10], [11, 22, 23, 24], [13], [14, 19, 20, 21]], from_n=16, to_n=10):
        down_matrix = paddle.zeros((from_n, to_n))
        for cnt, item in enumerate(down_sample_idx):
            factor = 1 / len(item)
            down_matrix[item, cnt] = factor
        return down_matrix

    def get_up_matrix(self, down_sample_idx=[[0, 15, 16, 17, 18], [1], [2, 3], [3, 4], [5, 6], [6, 7], [8, 9, 12], [10], [11, 22, 23, 24], [13], [14, 19, 20, 21]], from_n=10, to_n=16):
        up_matrix = paddle.zeros((from_n, to_n))
        for cnt, item in enumerate(down_sample_idx):
            up_matrix[cnt, item] = 1
        return up_matrix

    def forward(self, x, confidence):
        '''

        Args:
            x: b, 2, 300, 25, 1
            confidence:  b, 1, 300, 25, 1

        Returns:

        '''
        conf_x = x * confidence
        b, c, t, v, m = conf_x.shape
        conf_x = conf_x.squeeze(axis=-1).transpose((0, 3, 1, 2)).reshape((b, v*c, t)) # b, 50, 300

        feature = self.pose2feature(conf_x)
        # enc
        left_features = []
        for i in range(len(self.multi_scale_node_n)):
            feature = self.unet[f"enc_s{i+1}"](feature)
            left_features.append(feature)

            if i == 0:
                b, vc, t = feature.shape
                feature = paddle.matmul(feature.reshape((b, -1, 2, t)).transpose((0, 3, 2, 1)), self.down25_11_matrix).transpose((0, 3, 2, 1)).reshape((b, -1, t))
            elif i == 1:
                b, vc, t = feature.shape
                feature = paddle.matmul(feature.reshape((b, -1, 2, t)).transpose((0, 3, 2, 1)), self.down11_5_matrix).transpose((0, 3, 2, 1)).reshape((b, -1, t))
        # dec
        right_features = []
        for i in range(len(self.multi_scale_node_n)-1, -1, -1):
            if i == 2:
                feature = self.unet[f"dec_s{i+1}"](left_features[i])
            else:
                feature = self.unet[f"fuse_s{i+1}"](paddle.concat((left_features[i], feature), axis=-1))
                feature = self.unet[f"dec_s{i + 1}"](feature)

            right_features.append(feature)

            if i == 2:
                b, vc, t = feature.shape
                feature = paddle.matmul(feature.reshape((b, -1, 2, t)).transpose((0, 3, 2, 1)), self.up5_11_matrix).transpose((0, 3, 2, 1)).reshape((b, -1, t))
            elif i == 1:
                b, vc, t = feature.shape
                feature = paddle.matmul(feature.reshape((b, -1, 2, t)).transpose((0, 3, 2, 1)), self.up11_25_matrix).transpose((0, 3, 2, 1)).reshape((b, -1, t))
        right_features = right_features[::-1]

        # recognition
        recognitions = []
        for i in range(len(self.multi_scale_node_n)):
            recognitions.append(self.recognitions[i](right_features[i].reshape((b, -1))))
        recognitions = paddle.stack(recognitions, axis=1)
        recognitions = paddle.mean(recognitions, axis=1)
        return recognitions

class MSGCNDCT(nn.Layer):
    def __init__(self, in_len=30, multi_scale_node_n=[25, 11, 5], multi_scale_hidden_dim=[64, 128, 256], dropout_rate=0.2, down25_11 = [[0, 15, 16, 17, 18], [1], [2, 3], [3, 4], [5, 6], [6, 7], [8, 9, 12], [10], [11, 22, 23, 24], [13], [14, 19, 20, 21]], down11_5 = [[0, 1], [2, 3], [4, 5], [6, 7, 8], [6, 9, 10]]):
        super(MSGCNDCT, self).__init__()

        self.multi_scale_node_n = multi_scale_node_n
        self.multi_scale_hidden_dim = multi_scale_hidden_dim

        self.down25_11_matrix = self.get_down_matrix(down_sample_idx=down25_11, from_n=25, to_n=11)
        self.down11_5_matrix = self.get_down_matrix(down_sample_idx=down11_5, from_n=11, to_n=5)
        self.up11_25_matrix = self.get_up_matrix(down_sample_idx=down25_11, from_n=11, to_n=25)
        self.up5_11_matrix = self.get_up_matrix(down_sample_idx=down11_5, from_n=5, to_n=11)

        self.pose2feature = GraphConvBlock(in_len=in_len, out_len=multi_scale_hidden_dim[0], in_node_n=multi_scale_node_n[0]*2, out_node_n=multi_scale_node_n[0]*2, dropout_rate=dropout_rate, bias=True, residual=False)

        unet = nn.LayerDict()
        recognitions = nn.LayerList()
        for i, (n, h) in enumerate(zip(multi_scale_node_n, multi_scale_hidden_dim)):
            n = n * 2
            unet.update({
                f"enc_s{i+1}": nn.Sequential(
                    # ResGCB(in_len=h, out_len=h, in_node_n=n, out_node_n=n, dropout_rate=dropout_rate, bias=True, residual=True),
                    ResGCB(in_len=h, out_len=h, in_node_n=n, out_node_n=n, dropout_rate=dropout_rate, bias=True, residual=True),
                    ResGCB(in_len=h, out_len=h*2, in_node_n=n, out_node_n=n, dropout_rate=dropout_rate, bias=True, residual=False),
                ),
                f"dec_s{i + 1}": nn.Sequential(
                    ResGCB(in_len=h*2, out_len=h, in_node_n=n, out_node_n=n, dropout_rate=dropout_rate, bias=True, residual=False),
                    ResGCB(in_len=h, out_len=h, in_node_n=n, out_node_n=n, dropout_rate=dropout_rate, bias=True, residual=True),
                    # ResGCB(in_len=h, out_len=h, in_node_n=n, out_node_n=n, dropout_rate=dropout_rate, bias=True, residual=True),
                ),
            })
            if i != 2:
                unet.update({
                    f"fuse_s{i + 1}": GraphConvBlock(in_len=h*4, out_len=h*2, in_node_n=n, out_node_n=n,
                                                     dropout_rate=dropout_rate, bias=True, residual=False),
                })

            recognitions.append(nn.Sequential(
                nn.Linear(n * h, h // 2),
                nn.GELU(),
                nn.Linear(h // 2, h // 4),
                nn.GELU(),
                nn.Linear(h // 4, 30),
            ))
        self.unet = unet
        self.recognitions = recognitions


    def get_down_matrix(self, down_sample_idx=[[0, 15, 16, 17, 18], [1], [2, 3], [3, 4], [5, 6], [6, 7], [8, 9, 12], [10], [11, 22, 23, 24], [13], [14, 19, 20, 21]], from_n=16, to_n=10):
        down_matrix = paddle.zeros((from_n, to_n))
        for cnt, item in enumerate(down_sample_idx):
            factor = 1 / len(item)
            down_matrix[item, cnt] = factor
        return down_matrix

    def get_up_matrix(self, down_sample_idx=[[0, 15, 16, 17, 18], [1], [2, 3], [3, 4], [5, 6], [6, 7], [8, 9, 12], [10], [11, 22, 23, 24], [13], [14, 19, 20, 21]], from_n=10, to_n=16):
        up_matrix = paddle.zeros((from_n, to_n))
        for cnt, item in enumerate(down_sample_idx):
            up_matrix[cnt, item] = 1
        return up_matrix

    def forward(self, conf_data_dct):
        '''

        Args:
            conf_data_dct: b, 50, 30
        Returns:

        '''

        b, vc, t = conf_data_dct.shape

        feature = self.pose2feature(conf_data_dct)
        # enc
        left_features = []
        for i in range(len(self.multi_scale_node_n)):
            feature = self.unet[f"enc_s{i+1}"](feature)
            left_features.append(feature)

            if i == 0:
                b, vc, t = feature.shape
                feature = paddle.matmul(feature.reshape((b, -1, 2, t)).transpose((0, 3, 2, 1)), self.down25_11_matrix).transpose((0, 3, 2, 1)).reshape((b, -1, t))
            elif i == 1:
                b, vc, t = feature.shape
                feature = paddle.matmul(feature.reshape((b, -1, 2, t)).transpose((0, 3, 2, 1)), self.down11_5_matrix).transpose((0, 3, 2, 1)).reshape((b, -1, t))
        # dec
        right_features = []
        for i in range(len(self.multi_scale_node_n)-1, -1, -1):
            if i == 2:
                feature = self.unet[f"dec_s{i+1}"](left_features[i])
            else:
                feature = self.unet[f"fuse_s{i+1}"](paddle.concat((left_features[i], feature), axis=-1))
                feature = self.unet[f"dec_s{i + 1}"](feature)

            right_features.append(feature)

            if i == 2:
                b, vc, t = feature.shape
                feature = paddle.matmul(feature.reshape((b, -1, 2, t)).transpose((0, 3, 2, 1)), self.up5_11_matrix).transpose((0, 3, 2, 1)).reshape((b, -1, t))
            elif i == 1:
                b, vc, t = feature.shape
                feature = paddle.matmul(feature.reshape((b, -1, 2, t)).transpose((0, 3, 2, 1)), self.up11_25_matrix).transpose((0, 3, 2, 1)).reshape((b, -1, t))
        right_features = right_features[::-1]

        # recognition
        recognitions = []
        for i in range(len(self.multi_scale_node_n)):
            recognitions.append(self.recognitions[i](right_features[i].reshape((b, -1))))
        recognitions = paddle.stack(recognitions, axis=1)
        recognitions = paddle.mean(recognitions, axis=1)
        return recognitions

if __name__ == '__main__':
    m = MSGCNDCT()

    print(paddle.summary(m, input_size=(1, 50, 30)))
    x = paddle.randn((32, 50, 30))
    y = m(x)
    pass