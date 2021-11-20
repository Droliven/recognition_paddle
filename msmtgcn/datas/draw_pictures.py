#!/usr/bin/env python
# encoding: utf-8
'''
@project : t1111111111
@file    : draw_pictures.py
@author  : Levon
@contact : levondang@163.com
@ide     : PyCharm
@time    : 2021-11-19 18:06
'''


import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from matplotlib.pyplot import MultipleLocator
import cv2


def draw_pic_single_2d(mydata, I, J, LR, full_path):
    # 22, 3, XZY

    x = mydata[:, 0]
    y = mydata[:, 1]
    x *= -1000
    y *= -1000

    plt.figure(figsize=(6, 6))

    # Make connection matrix
    for i in np.arange(len(I)):
        x, y = [np.array([mydata[I[i], j], mydata[J[i], j]]) for j in range(2)]
        plt.plot(x, y, lw=2, color='g' if LR[i] else 'b')

    plt.xlim((-800, 800))
    plt.ylim((-1000, 1000))
    # 设置坐标轴名称
    plt.xlabel('x')
    plt.ylabel('y')
    # 设置坐标轴刻度
    my_x_ticks = np.arange(-800, 800, 200)
    my_y_ticks = np.arange(-1000, 1000, 200)
    plt.xticks(my_x_ticks)
    plt.yticks(my_y_ticks)
    plt.grid(False)

    plt.savefig(full_path)
    plt.close(1)

def draw_multi_seqs_2d(seqs, gt_cnt=3, I=[], J=[], LR=[], t_his=25, full_path="", x_period=[-800, 800], y_period=[1000, 1000], z_period=[-1000, 1000]):
    n, t, v, c = seqs.shape  # n, 125, 17, 2

    gts = seqs[:gt_cnt]
    preds = seqs[gt_cnt:]

    # (250, 40, 40) #FA2828 红
    # (245, 125, 125) #F57D7D 粉
    # (11, 11, 11) #0B0B0B 黑色
    # (180, 180, 180) #B4B4B4 灰色
    # blue_left = (0, 0, 205) # #0000CD
    # blue_right = (100, 149, 237)  #6495ED

    plt.figure(figsize=(int((x_period[1]-x_period[0]) / 1000 * t), int((z_period[1]-z_period[0]) / 1000 * n)))  # 只有设置为相等的值，才能保证坐标轴等间隔不会变形
    plt.xlabel('x')
    plt.ylabel('z')

    # 设置坐标轴刻度
    plt.xlim(x_period[0], x_period[1] + ((t - 1) * (x_period[1] - x_period[0])))
    plt.ylim(z_period[0] - ((n - 1) * (z_period[1] - z_period[0])), z_period[1])

    my_x_ticks = np.arange(x_period[0], x_period[1] + (t - 1) * (x_period[1] - x_period[0]), 1000)
    my_y_ticks = np.arange(z_period[0] - ((n - 1) * (z_period[1] - z_period[0])), z_period[1], 1000)
    plt.xticks(my_x_ticks)
    plt.yticks(my_y_ticks)

    plt.grid(False)

    draw_cnt = 0
    for gtidx in range(gts.shape[0]):
        draw_gt = gts[gtidx]  # t, v, c
        for tidx in range(draw_gt.shape[0]):
            pose = draw_gt[tidx]  # v, c
            pose[:, 0] = pose[:, 0] + (tidx * (x_period[1] - x_period[0]))
            pose[:, 1] = pose[:, 1] - (draw_cnt * (z_period[1] - z_period[0]))

            if tidx < t_his:
                # plt.scatter(pose[:, 0], pose[:, 1], c='k', linewidths=1)
                for i in np.arange(len(I)):
                    x, y = [np.array([pose[I[i], j], pose[J[i], j]]) for j in range(2)]
                    plt.plot(x, y, lw=1, color='#0B0B0B' if LR[i] else '#B4B4B4')
            else:
                # plt.scatter(pose[:, 0], pose[:, 1], c='b', linewidths=1)
                for i in np.arange(len(I)):
                    x, y = [np.array([pose[I[i], j], pose[J[i], j]]) for j in range(2)]
                    plt.plot(x, y, lw=1, color='#0000CD' if LR[i] else '#6495ED')

        draw_cnt += 1

    for predidx in range(preds.shape[0]):
        draw_pred = preds[predidx]  # t, v, c
        for tidx in range(draw_pred.shape[0]):
            pose = draw_pred[tidx]  # v, c
            pose[:, 0] = pose[:, 0] + (tidx * (x_period[1] - x_period[0]))
            pose[:, 1] = pose[:, 1] - (draw_cnt * (z_period[1] - z_period[0]))

            if tidx < t_his:
                # plt.scatter(pose[:, 0], pose[:, 1], c='k', linewidths=1)
                for i in np.arange(len(I)):
                    x, y = [np.array([pose[I[i], j], pose[J[i], j]]) for j in range(2)]
                    plt.plot(x, y, lw=1, color='#0B0B0B' if LR[i] else '#B4B4B4')
            else:
                # plt.scatter(pose[:, 0], pose[:, 1], c='b', linewidths=1)
                for i in np.arange(len(I)):
                    x, y = [np.array([pose[I[i], j], pose[J[i], j]]) for j in range(2)]
                    plt.plot(x, y, lw=1, color='#FA2828' if LR[i] else '#F57D7D')

        draw_cnt += 1

    plt.savefig(full_path)
    plt.close(1)

def draw_multi_poses_2d(seqs, gt_cnt=3, I=[], J=[], LR=[], tidx=2, t_his=25, full_path="", x_period=[-1100, 1800], y_period=[-1600, 2700], z_period=[-200, 2100]):
    n, v, c = seqs.shape  # n, 17, 2

    gts = seqs[:gt_cnt, :, :]
    preds = seqs[gt_cnt:, :, :]

    # (250, 40, 40) #FA2828 红
    # (245, 125, 125) #F57D7D 粉
    # (11, 11, 11) #0B0B0B 黑色
    # (180, 180, 180) #B4B4B4 灰色
    # blue_left = (0, 0, 205) # #0000CD
    # blue_right = (100, 149, 237)  #6495ED

    plt.figure(figsize=(int((x_period[1]-x_period[0]) / 1000), int((z_period[1]-z_period[0]) / 1000 * n)))  # 只有设置为相等的值，才能保证坐标轴等间隔不会变形

    plt.xlabel('x')
    plt.ylabel('z')

    # 设置坐标轴刻度
    plt.xlim(x_period[0], x_period[1])
    plt.ylim(z_period[0] - ((n - 1) * (z_period[1] - z_period[0])), z_period[1])

    my_x_ticks = np.arange(x_period[0], x_period[1], 1000)
    my_y_ticks = np.arange(z_period[0] - ((n - 1) * (z_period[1] - z_period[0])), z_period[1], 1000)
    plt.xticks(my_x_ticks)
    plt.yticks(my_y_ticks)

    plt.grid(False)

    draw_cnt = 0
    for gtidx in range(gts.shape[0]):
        pose = gts[gtidx]  # v, c
        pose[:, 1] = pose[:, 1] - (draw_cnt * (z_period[1] - z_period[0]))

        if tidx < t_his:
            # plt.scatter(pose[:, 0], pose[:, 1], c='k', linewidths=1)
            for i in np.arange(len(I)):
                x, y = [np.array([pose[I[i], j], pose[J[i], j]]) for j in range(2)]
                plt.plot(x, y, lw=1, color='#0B0B0B' if LR[i] else '#B4B4B4')
        else:
            # plt.scatter(pose[:, 0], pose[:, 1], c='b', linewidths=1)
            for i in np.arange(len(I)):
                x, y = [np.array([pose[I[i], j], pose[J[i], j]]) for j in range(2)]
                plt.plot(x, y, lw=1, color='#0000CD' if LR[i] else '#6495ED')

        draw_cnt += 1

    for predidx in range(preds.shape[0]):
        pose = preds[predidx]  # v, c
        pose[:, 1] = pose[:, 1] - (draw_cnt * (z_period[1] - z_period[0]))

        if tidx < t_his:
            # plt.scatter(pose[:, 0], pose[:, 1], c='k', linewidths=1)
            for i in np.arange(len(I)):
                x, y = [np.array([pose[I[i], j], pose[J[i], j]]) for j in range(2)]
                plt.plot(x, y, lw=1, color='#0B0B0B' if LR[i] else '#B4B4B4')
        else:
            # plt.scatter(pose[:, 0], pose[:, 1], c='r', linewidths=1)
            for i in np.arange(len(I)):
                x, y = [np.array([pose[I[i], j], pose[J[i], j]]) for j in range(2)]
                plt.plot(x, y, lw=1, color='#FA2828' if LR[i] else '#F57D7D')

        draw_cnt += 1

    plt.savefig(full_path)
    plt.close(1)


if __name__ == "__main__":

    hex_color = "#FA2828"
    rgb = Hex_to_RGB(hex_color)
    new_hex = RGB_to_Hex("250,40,40")
    end_rgb = map(lambda x: int(x), Hex_to_RGB(hex_color).split(","))
    gradient = color_gradient_hex()
    pass

