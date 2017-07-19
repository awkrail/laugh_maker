import numpy as np
import pandas as pd
import random
import csv
from chainer.datasets import tuple_dataset

'''
データを保持して加工するクラス
80% -> 訓練, テストデータとして利用
20% -> validationのためのデータとして利用
MatrixSensorData->84次元のデータ
OneDimSensorData->平均して, 5秒ごとにサンプリングした時のデータ
'''


class MatrixSensorData(object):
    """
    データを加工して, chainerのためのデータを作成するクラス
    プロパティ:
        self.ary_0 -> 笑っていない時のデータ
        self.ary_1 -> 笑っている時のデータ
    """
    def __init__(self):
        self.ary_0 = []
        self.ary_1 = []
        self.ary0_50dim = []
        self.ary1_50dim = []

    def load_csv(self):
        self.ary_0 = pd.read_csv('84Raw/0.csv', header=None)
        self.ary0_50dim = self.make_50dim_array(self.ary_0, 0)
        self.ary_1 = pd.read_csv('84Raw/1.csv', header=None)
        self.ary1_50dim = self.make_50dim_array(self.ary_1, 1)

    def divide_train_and_validation(self):
        # シャッフルして, 訓練データとテストデータに分割する。
        all_ary = self.ary0_50dim + self.ary1_50dim
        all_ary_len = len(all_ary)
        random.shuffle(all_ary)

        pannel_data = [tuple_data[0] for tuple_data in all_ary]
        answer_data = [tuple_data[1] for tuple_data in all_ary]

        # 訓練データと, テストデータへ分ける
        threshold = np.int32(all_ary_len/4*5)
        train = tuple_dataset.TupleDataset(pannel_data[0:threshold], answer_data[0:threshold])
        test = tuple_dataset.TupleDataset(pannel_data[threshold:], answer_data[threshold:])

        return train, test

    @staticmethod
    def make_50dim_array(ary_1dim, answer):
        tmp_ary = []
        tmp_50dim = []
        iter_count = 0
        for _, data in ary_1dim.iterrows():
            if iter_count == 50:
                tmp_50dim.append((np.array(tmp_ary, dtype=np.float32), np.int32(answer)))
                tmp_ary = []
                iter_count = 0
                continue
            tmp_ary.append(data.values.reshape(6, 14))
            iter_count += 1

        return tmp_50dim


class OneDimSensorData(object):
    """
    センサーデータを訓練データとテストデータにわけるためのクラス
    :param
        self.ary0or1 -> 0or1のセンサーデータ(tuple, データ, ラベル)
    :return
        divide_train_and_test
            訓練データ, 訓練ラベル, テストデータ, テストラベル
    """
    def __init__(self):
        self.ary0 = []
        self.ary1 = []
        self.all = []

    def load_csv(self):
        with open('confused_data/Raw/0.csv', 'r') as f:
            self.ary0 = [(np.array(row[0:50], dtype=np.float32), np.array(0, dtype=np.int32))
                         for row in csv.reader(f)]

        with open('confused_data/Raw/1.csv', 'r') as f:
            self.ary1 = [(np.array(row[0:50], dtype=np.float32), np.array(1, dtype=np.int32))
                         for row in csv.reader(f)]

        self.all = self.ary0 + self.ary1

    def shuffle(self):
        random.shuffle(self.all)

    def divide_train_and_test(self):
        threshold = int(len(self.all)/5*4)
        train = []
        train_label = []
        test = []
        test_label = []
        for i, data in enumerate(self.all):
            if i < threshold:
                train.append(data[0])
                train_label.append(data[1])
            else:
                test.append(data[0])
                test_label.append(data[1])

        # for recall rate
        one_label_count = 0
        # import ipdb; ipdb.set_trace()

        for label in test_label:
            if label == 1:
                one_label_count += 1

        train = np.array(train, dtype=np.float32)
        train_label = np.array(train_label, dtype=np.int32)
        test = np.array(test, dtype=np.float32).reshape(10, 741, 50) # 全体-threshold分
        test_label = np.array(test_label, dtype=np.int32).reshape(10, 741)

        return train, train_label, test, test_label, one_label_count







