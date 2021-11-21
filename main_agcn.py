#!/usr/bin/env python
# encoding: utf-8
'''
@project : PaddleVideo
@file    : main_levon_recognition.py
@author  : Levon
@contact : levondang@163.com
@ide     : PyCharm
@time    : 2021-11-18 21:40
'''
import paddle
import os
import numpy as np

PLACE = paddle.CUDAPlace(0)

from baselines.agcn.runs import Runner

runner = Runner()
runner.run()
