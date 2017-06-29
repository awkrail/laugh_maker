from sklearn import svm
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import csv


class Ary4SvmClass(object):
    def __init__(self):
        self.ary0 = []
        self.answer0 = []
        self.ary1 = []
        self.answer1 = []

    def load_csv(self):
        # aryで掴むにはcsvモジュールのほうがやりやすい。
        with open('confused_data/Raw/0.csv', 'r') as f:
            self.ary0 = [row[0:50] for row in csv.reader(f)]
            self.answer0 = [0 for _ in range(len(self.ary0))]

        with open('confused_data/Raw/1.csv', 'r') as f:
            self.ary1 = [row for row in csv.reader(f)]
            self.answer1 = [1 for _ in range(len(self.ary1))]


# 適合率を測定します。1のデータだけ取り出す。
class HighDimension4SvmClass(object):
    """
        適合率を測定する。
        self.ary1 -> わらっているときのデータ
        self.answer1 -> 1というデータセット
    """
    def __init__(self):
        self.ary0 = []
        self.ary1 = []
        self.answer0 = []
        self.answer1 = []

    def load_csv(self):
        with open('84Raw/0.csv', 'r') as f:
            self.ary0 = [row for row in csv.reader(f)]

        with open('84Raw/1.csv', 'r') as f:
            self.ary1 = [row for row in csv.reader(f)]

        self.ary0 = self.pd2ary(self.ary0)
        self.answer0 = [0 for _ in range(len(self.ary0))]
        self.ary1 = self.pd2ary(self.ary1)
        self.answer1 = [1 for _ in range(len(self.ary1))]

    @staticmethod
    def pd2ary(self_ary):
        count = 0
        tmp_ary = []
        return_ary = []
        for row in self_ary:
            if count == 50:
                return_ary.append(np.array(tmp_ary, dtype=np.float32))
                tmp_ary = []
                count = 0
                continue
            mean = float(np.array(row, dtype=np.float32).mean()) # 平均を計算
            tmp_ary.append(mean)
            count += 1

        return return_ary

    def calculate_recall_rate(self, clf):
        recall = 0
        question_recall = len(self.answer1)

        for data, test_answer in zip(self.ary1, self.answer1):
            answer = clf.predict([data])
            if answer == test_answer:
                recall += 1

        print('recall rate: ', recall / question_recall)

    def calculate_accuracy_rate(self, clf):
        all_ary = self.ary0 + self.ary1
        all_answer = self.answer0 + self.answer1
        accuracy = 0
        question_acc = len(all_ary)

        # import ipdb; ipdb.set_trace()

        for data, test_answer in zip(all_ary, all_answer):
            answer = clf.predict([data])
            if answer == test_answer:
                accuracy += 1
        print('accuracy rate: ', accuracy / question_acc)

def main():
    raw_data = Ary4SvmClass()
    raw_data.load_csv()

    # 全部のデータを獲得する
    all_data = np.array(raw_data.ary0 + raw_data.ary1, dtype=np.float32)
    label_data = np.array(raw_data.answer0 + raw_data.answer1, dtype=np.float32)

    # 学習用, テスト用にデータを分ける
    x_train, x_test, y_train, y_test = train_test_split(all_data, label_data, test_size=0.33, random_state=30)

    # import ipdb; ipdb.set_trace()
    clf = svm.SVC()
    clf.fit(x_train, y_train)

    # 正答率
    accuracy = 0
    question_num = len(x_test)

    # 適合率測定
    recall = 0
    question_recall = len(raw_data.answer1)

    """
    for test_data, test_answer in zip(x_test, y_test):
        answer = clf.predict([test_data])
        if test_answer == answer:
            accuracy += 1

        if answer == test_answer and test_answer == 1:
            recall += 1

    print('accuracy rate:', accuracy / question_num)
    print('recall rate: ', recall / question_recall)
    """
    # 日常データで測定(適合率のみ)
    daily_data = HighDimension4SvmClass()
    daily_data.load_csv()
    daily_data.calculate_accuracy_rate(clf)
    daily_data.calculate_recall_rate(clf)


if __name__ == '__main__':
    main()
