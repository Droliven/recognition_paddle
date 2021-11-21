#!/usr/bin/env python
# encoding: utf-8
'''
@project : PaddleVideo
@file    : runner.py
@author  : Levon
@contact : levondang@163.com
@ide     : PyCharm
@time    : 2021-11-18 20:28
'''
import paddle
import os
import numpy as np
from visualdl import LogWriter
from paddle.io import DataLoader
from paddle.optimizer import Adam
from pprint import pprint
import paddle.nn.functional as F
import pandas as pd

from baselines.agcn.datas import SkeletonDataset
from baselines.agcn.nets import AGCN
from baselines.agcn.cfgs import Config
from baselines.agcn.runs import CrossEntropyLoss


class Runner():
    def __init__(self):
        super(Runner, self).__init__()
        # 参数
        self.start_epoch = 1
        self.best_accuracy = 1e15

        self.cfg = Config()
        print("\n================== Configs =================")
        pprint(vars(self.cfg), indent=4)
        print("==========================================\n")
        with open(os.path.join(self.cfg.ckpt_base_dir, "config.txt"), 'w', encoding='utf-8') as f:
            f.write(str(self.cfg.__dict__))


        self.model = AGCN()
        # print(paddle.summary(self.model, (1, 2, 300, 25, 1)))
        self.scheduler = paddle.optimizer.lr.StepDecay(learning_rate=self.cfg.lr, step_size=5, gamma=0.9, verbose=True)
        self.optimizer = Adam(parameters=self.model.parameters(), learning_rate=self.scheduler)

        train_dataset = SkeletonDataset(file_path=self.cfg.train_data_path, label_path=self.cfg.train_label_path, mode="train", segment=self.cfg.segment, repeat=self.cfg.repeat)
        self.train_loader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=self.cfg.train_batch_size, num_workers=self.cfg.num_workers, drop_last=True)

        test_dataset = SkeletonDataset(file_path=self.cfg.test_data_path, label_path=None, mode="test", segment=self.cfg.segment, repeat=self.cfg.repeat)
        self.test_loader = DataLoader(dataset=test_dataset, shuffle=False, batch_size=self.cfg.test_batch_size, num_workers=self.cfg.num_workers, drop_last=False)

        self.summary = LogWriter(logdir=self.cfg.ckpt_base_dir)


    def save(self, checkpoint_path, epoch, best_err, curr_err):
        state_dict = {
            "epoch": epoch,
            "lr": self.optimizer.get_lr(),
            "best_err": best_err,
            "curr_err": curr_err,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        paddle.save(state_dict, checkpoint_path)


    def load(self, checkpoint_path):
        state = paddle.load(checkpoint_path)
        self.model.set_state_dict(state["model"])
        self.optimizer.set_state_dict(state["optimizer"])
        # self.lr = state["lr"]
        # self.start_epoch = state["epoch"] + 1
        best_err = state['best_err']
        curr_err = state["curr_err"]
        print("load from epoch {}, lr {}, curr_avg {}, best_avg {}.".format(state["epoch"], state["lr"], curr_err, best_err))


    def train(self, epoch):
        self.model.train()
        average_all_loss = 0

        generator_len = len(self.train_loader)
        for i, (data, label) in enumerate(self.train_loader):
            # b, 2, 300, 25, 1
            self.global_step = (epoch - 1) * generator_len + i + 1
            b = data.shape[0]
            # skip the last batch if only have one sample for batch_norm layers
            if b == 1:
                continue
            self.global_step = (epoch - 1) * generator_len + i + 1

            out = self.model(data) # b, 30
            loss = F.cross_entropy(out, label)
            self.summary.add_scalar(tag=f"loss", value=loss, step=self.global_step)

            loss.backward()
            self.optimizer.step()
            self.optimizer.clear_grad()

            average_all_loss += loss.numpy()[0]

        average_all_loss /= generator_len
        self.summary.add_scalar(tag=f"loss_avg", value=average_all_loss, step=epoch)
        return average_all_loss


    def test(self, epoch):
        self.model.eval()
        generator_len = len(self.test_loader)

        all_recognition = np.zeros((generator_len, 2)).astype(np.int64) - 1
        all_recognition[:, 0] = np.arange(0, generator_len).astype(np.int64)
        for i, data in enumerate(self.test_loader):  # # b, [repeats], 2, 100, 25, 1
            with paddle.no_grad():
                all_outs = []
                for h in range(self.cfg.repeat):
                    ipt = data[:, h]
                    out = self.model(ipt)
                    all_outs.append(out)
                all_outs = paddle.mean(paddle.stack(all_outs, axis=0), axis=0) # b, 30
                prob = F.softmax(all_outs)
                recognition = paddle.argmax(prob, axis=1).numpy()[0]
                all_recognition[i][1] = recognition

        # 保存为 csv
        result = pd.DataFrame(all_recognition, columns=['sample_index', 'predict_category'], index=None)
        result.to_csv(os.path.join(self.cfg.ckpt_base_dir, f"submission_{epoch}.csv"), index=False, encoding="utf-8", line_terminator="\n")


    def run(self):
        for epoch in range(self.start_epoch, self.cfg.n_epoch + 1):

            # if epoch % 5 == 0:
            #     self.lr = lr_decay(self.optimizer, self.lr, self.cfg.lr_decay)
            self.summary.add_scalar("LR", self.optimizer.get_lr(), epoch)

            average_all_loss = self.train(epoch)
            self.scheduler.step()

            print("Epoch {}, average_all_loss {:.4f}".format(epoch, average_all_loss))
            if average_all_loss < self.best_accuracy:
                self.best_accuracy = average_all_loss
                self.save(os.path.join(self.cfg.ckpt_base_dir, "models", 'epoch{}_err{:.4f}.pth'.format(epoch, average_all_loss)), epoch, self.best_accuracy, average_all_loss)

            self.save(os.path.join(self.cfg.ckpt_base_dir, "models", 'last.pth'), epoch, self.best_accuracy, average_all_loss)

            if epoch % 10 == 0:
                self.test(epoch)


if __name__ == '__main__':
    r = Runner()
    # r.save(checkpoint_path=os.path.join(r.cfg.ckpt_base_dir, "models", 'epoch{}_err{:.4f}.pth'.format(-1, 5.02)), epoch=-1, best_err=3.01, curr_err=5.02)
    # r.load(checkpoint_path=os.path.join(r.cfg.ckpt_base_dir, "models", '{}_err{:.4f}.pth'.format(-1, 5.02)))
    # r.test()

    r.run()
    pass
