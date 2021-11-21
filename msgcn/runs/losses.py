#!/usr/bin/env python
# encoding: utf-8
'''
@project : PaddleVideo
@file    : losses.py
@author  : Levon
@contact : levondang@163.com
@ide     : PyCharm
@time    : 2021-11-18 20:30
'''
import paddle
import paddle.nn.functional as F

class CrossEntropyLoss(paddle.nn.Layer):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        pass

    """Cross Entropy Loss."""
    def forward(self, score, labels):
        """Forward function.
        Args:
            score (paddle.Tensor): The class score.
            labels (paddle.Tensor): The ground truth labels.
            kwargs: Any keyword argument to be used to calculate
                CrossEntropy loss.
        Returns:
            loss (paddle.Tensor): The returned CrossEntropy loss.
        """
        loss = F.cross_entropy(score, labels)
        return loss

