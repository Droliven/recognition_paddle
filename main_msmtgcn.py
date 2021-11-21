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

from msmtgcn.runs import Runner, RunnerDCT

model_path = os.path.join(r"E:\PythonWorkspace\recognition_paddle\ckpt\msgcndct\models", "epoch400_top1_0.4437.pth")

# runner = Runner(exp_name="msgcn")
runner = RunnerDCT(exp_name="msgcndct")

runner.load(model_path)
# top1 = runner.val_repeat_top1(epoch=-1)
runner.test(-1)

# runner.run()