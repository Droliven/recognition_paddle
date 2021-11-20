#!/usr/bin/env python
# encoding: utf-8
'''
@project : t1111111111
@file    : layers.py
@author  : Levon
@contact : levondang@163.com
@ide     : PyCharm
@time    : 2021-11-19 19:14
'''

import paddle
import paddle.nn as nn
import math

class GraphConv(nn.Layer):
    """
        adapted from : https://github.com/tkipf/gcn/blob/92600c39797c2bfb61a508e52b88fb554df30177/gcn/layers.py#L132
        """

    def __init__(self, in_len, out_len, in_node_n=66, out_node_n=66, bias=True):
        super(GraphConv, self).__init__()
        self.in_len = in_len
        self.out_len = out_len
        self.in_node_n = in_node_n
        self.out_node_n = out_node_n

        self.weight = paddle.create_parameter(shape=[in_len, out_len], dtype=paddle.float32)
        self.att = paddle.create_parameter(shape=[in_node_n, out_node_n], dtype=paddle.float32)

        if bias:
            self.bias = paddle.create_parameter(shape=[out_len], dtype=paddle.float32)


    def forward(self, input):
        '''
        b, cv, t
        '''

        features = paddle.matmul(input, self.weight)  # 35 -> 256
        output = paddle.matmul(features.transpose((0, 2, 1)), self.att).transpose((0, 2, 1))  # 66 -> 66

        if self.bias is not None:
            output = output + self.bias

        return output


class GraphConvBlock(nn.Layer):
    def __init__(self, in_len, out_len, in_node_n, out_node_n, dropout_rate=0, leaky=0.1, bias=True, residual=False):
        super(GraphConvBlock, self).__init__()
        self.dropout_rate = dropout_rate
        self.resual = residual

        self.out_len = out_len

        self.gcn = GraphConv(in_len, out_len, in_node_n=in_node_n, out_node_n=out_node_n, bias=bias)
        # self.bn = nn.BatchNorm1D(out_node_n * out_len)
        self.bn = nn.BatchNorm2D(2)
        # self.act = nn.LeakyReLU(leaky)
        self.act = nn.Tanh()
        # self.act = nn.ReLU()
        # self.act = nn.GELU()
        if self.dropout_rate > 0:
            self.drop = nn.Dropout(dropout_rate)

    def forward(self, input):
        '''

        Args:
            input: b, cv, t

        Returns:

        '''
        x = self.gcn(input)
        b, vc, t = x.shape
        # x = self.bn(x.reshape((b, -1))).reshape((b, vc, t))
        x = self.bn(x.reshape((b, -1, 2, t)).transpose((0, 2, 1, 3))).transpose((0, 2, 1, 3)).reshape((b, vc, t))
        x = self.act(x)
        if self.dropout_rate > 0:
            x = self.drop(x)

        if self.resual:
            return x + input
        else:
            return x


class ResGCB(nn.Layer):
    def __init__(self, in_len, out_len, in_node_n, out_node_n, dropout_rate=0, leaky=0.1, bias=True, residual=False, cat_left=False):
        super(ResGCB, self).__init__()
        self.cat_left = cat_left
        self.resual = residual

        self.gcb1 = GraphConvBlock(in_len, in_len, in_node_n=in_node_n, out_node_n=in_node_n, dropout_rate=dropout_rate, bias=bias, residual=False)
        self.gcb2 = GraphConvBlock(in_len, out_len, in_node_n=in_node_n, out_node_n=out_node_n, dropout_rate=dropout_rate, bias=bias, residual=False)
        if cat_left:
            self.crop = GraphConvBlock(in_len, in_len, in_node_n=in_node_n*2, out_node_n=in_node_n, dropout_rate=dropout_rate, bias=bias, residual=False)


    def forward(self, input, left=None):
        '''

        Args:
            x: B,CV,T

        Returns:

        '''
        if self.cat_left:
            assert left is not None
            cat = paddle.concat((input, left), axis=-2)
            input = self.crop(cat)

        x = self.gcb1(input)
        x = self.gcb2(x)

        if self.resual:
            return x + input
        else:
            return x

if __name__ == '__main__':
    m = ResGCB(in_len=10, out_len=25, in_node_n=48, out_node_n=48, dropout_rate=0.1, bias=True, residual=False, cat_left=False)

    x = paddle.randn((4, 48, 10))
    y = m(x)
    pass
