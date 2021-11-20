#!/usr/bin/env python
# encoding: utf-8
'''
@project : t1111111111
@file    : main_msmtgcn.py
@author  : Levon
@contact : levondang@163.com
@ide     : PyCharm
@time    : 2021-11-20 01:34
'''

import paddle
import os
import numpy as np

PLACE = paddle.CUDAPlace(0)  # 5, 7

from msmtgcn.runs import Runner

model_path = os.path.join(r"E:\PythonWorkspace\PaddleVideo\t1111111111\outputs\msgcn", "last.pth")

runner = Runner(exp_name="msmrgcn")

# runner.load(model_path)
# top1 = runner.val_repeat_top1(epoch=-1)
# runner.test(-1)

runner.run()