# -*- coding: utf-8 -*-
import numpy as np
import csv
import chainer
import chainer.functions as F
import chainer.links as L
import sys
import matplotlib.pyplot as plt

"""
データを実際に予測するときに利用するクラス
    :param
    predict_ary: csvからデータを取ってきて, 入力と正解をタプル形式で配列で持つクラス
    plt_ary: matplotlibで表示するためにもつ配列
    model: chainerでつくったモデルのインスタンスを格納
    one_count: 正解データで"1"のラベルのもの

    性能はrecall_rateで測定(TP/TP+TN)
"""


class PredictorClass(object):
    def __init__(self):
        self.predict_ary = []
        self.plt_ary = []
        self.model = None
        self.one_count = None

    def load_model(self, m_cls, path):
        self.model = m_cls
        chainer.serializers.load_npz(path, m_cls)

    def load_csv(self, path):
        with open(path, 'r') as f:
            self.predict_ary = [(np.array(row[:-1], dtype=np.float32), int(row[-1])) for row in csv.reader(f)]

    def predict(self):
        ans_count = 0
        one_count = 0
        question_num = len(self.predict_ary)

        for data in self.predict_ary:
            reshape_np = data[0].reshape(1, 50)
            answer = data[1]
            y = np.argmax(F.softmax(model(reshape_np)).data)
            if answer == 1:
                one_count += 1
            if y == answer:
                ans_count += 1
                self.plt_ary.append((self.model(reshape_np).data, y, 1))
            else:
                self.plt_ary.append((self.model(reshape_np).data, y, 0))

        accuracy = ans_count / question_num
        self.one_count = one_count

        return accuracy

    def plot(self):
        x_ary0 = []
        y_ary0 = []
        x_ary1 = []
        y_ary1 = []

        incorrect_x0 = []
        incorrect_y0 = []

        incorrect_x1 = []
        incorrect_y1 = []

        for dt in self.plt_ary:
            if dt[2] == 0:
                if dt[1] == 0:
                    incorrect_x0.append(dt[0][0][0])
                    incorrect_y0.append(dt[0][0][1])
                    continue
                else:
                    incorrect_x1.append(dt[0][0][0])
                    incorrect_y1.append(dt[0][0][1])
                    continue
            if dt[1] == 0:
                x_ary0.append(dt[0][0][0])
                y_ary0.append(dt[0][0][1])
            else:
                x_ary1.append(dt[0][0][0])
                y_ary1.append(dt[0][0][1])

        np_plt_x0 = np.array(x_ary0, dtype=np.float32)
        np_plt_y0 = np.array(y_ary0, dtype=np.float32)

        print('0->1: ', len(incorrect_x0))
        print('1->0: ', len(incorrect_x1))
        print('1->1: ', len(x_ary1))
        print('0->0: ', len(x_ary0))
        print('recall rate: ', len(x_ary1) / self.one_count)
        print('accuracy: ', (len(x_ary1)+len(x_ary0)) / len(self.plt_ary))

        np_plt_x1 = np.array(x_ary1, dtype=np.float32)
        np_plt_y1 = np.array(y_ary1, dtype=np.float32)

        np_incorrectx0 = np.array(incorrect_x0, dtype=np.float32)
        np_incorrecty0 = np.array(incorrect_y0, dtype=np.float32)

        np_incorrectx1 = np.array(incorrect_x1, dtype=np.float32)
        np_incorrecty1 = np.array(incorrect_y1, dtype=np.float32)

        plt.plot(np_plt_x1, np_plt_y1, 'o', color='r', alpha=0.5)
        plt.plot(np_plt_x0, np_plt_y0, 'o', color='b', alpha=0.5)
        plt.plot(np_incorrectx0, np_incorrecty0, 'o', color='g', alpha=0.5)
        plt.plot(np_incorrectx1, np_incorrecty1, 'o', color='m', alpha=0.5)
        plt.show()
