from sklearn import svm
import numpy as np
from sklearn.externals import joblib
import csv


class Daily50dimClass(object):
    def __init__(self):
        self.daily_ary = []
        self.label_ary = []
        self.recall_num = 0
        self.num_0 = 0

    def load_csv(self):
        with open('validation.csv', 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                self.daily_ary.append(row[:-1])
                self.label_ary.append(int(row[-1]))
                if int(row[50]) == 1:
                    self.recall_num += 1
                else:
                    self.num_0 += 1

        self.daily_ary = np.array(self.daily_ary, dtype=np.float32)


def main():
    # loading dataset
    daily_data = Daily50dimClass()
    daily_data.load_csv()

    # loading model
    clf = joblib.load('clf.pkl')

    accuracy = 0
    question_num = len(daily_data.daily_ary)

    recall = 0
    answer_one_num = daily_data.recall_num
    answer_zero_num = daily_data.num_0

    for data, label in zip(daily_data.daily_ary, daily_data.label_ary):
        answer = clf.predict([data])
        if answer == label:
            accuracy += 1
            if answer == 1:
                recall += 1

    # import ipdb; ipdb.set_trace()

    print('question_num: ', question_num)
    print('answer_one_num: ', answer_one_num)
    print('answer_zero_num: ', answer_zero_num)
    print('daily data accuracy: ', accuracy / question_num)
    print('recall rate: ', recall / answer_one_num)
    print(recall)


if __name__ == '__main__':
    main()




