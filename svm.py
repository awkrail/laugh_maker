from sklearn import svm
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.externals import joblib
import csv
import os
import random

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


class DailyDataLoader(object):
    def __init__(self):
        self.ary0 = []
        self.answer0 = []
        self.ary1 = []
        self.answer1 = []

    def load_csv(self):
        directories = os.listdir('train_daily_many')
        for directory in directories:
            with open('train_daily_many/' + directory, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    if int(row[-1]) == 0:
                        self.ary0.append((np.array(row[:-1], dtype=np.float32), 0))
                    else:
                        self.ary1.append((np.array(row[:-1], dtype=np.float32), 1))

    def shuffle_ary(self):
        ary1_length = len(self.ary1)
        shuffled_ary0 = random.sample(self.ary0, ary1_length)

        self.ary0 = [tuple_data[0] for tuple_data in shuffled_ary0]
        self.answer0 = [0 for _ in range(len(self.ary0))]
        self.ary1 = [tuple_data[0] for tuple_data in self.ary1]
        self.answer1 = [1 for _ in range(len(self.ary1))]


def main():
    raw_data = DailyDataLoader()
    raw_data.load_csv()
    raw_data.shuffle_ary()

    # 全部のデータを獲得する
    all_data = np.array(raw_data.ary0 + raw_data.ary1, dtype=np.float32)
    label_data = np.array(raw_data.answer0 + raw_data.answer1, dtype=np.int32)

    # 学習用, テスト用にデータを分ける
    x_train, x_test, y_train, y_test = train_test_split(all_data, label_data, test_size=0.4, random_state=30)

    # import ipdb; ipdb.set_trace()
    clf = svm.SVC()
    clf.fit(x_train, y_train)

    # 正答率
    accuracy = 0
    question_num = len(x_test)

    # 適合率測定
    recall = 0
    question_recall = len(raw_data.answer1)

    for test_data, test_answer in zip(x_test, y_test):
        answer = clf.predict([test_data])
        if test_answer == answer:
            accuracy += 1

        if answer == test_answer and test_answer == 1:
            recall += 1

    print('accuracy rate:', accuracy / question_num)
    print('recall rate: ', recall / question_recall)

    # うまくいかないので, validation_src/以下で検証。モデルをシリアライズする。
    joblib.dump(clf, 'clf.pkl')


if __name__ == '__main__':
    main()
