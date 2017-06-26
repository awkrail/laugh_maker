import numpy as np
import csv
import chainer
import chainer.functions as F
import chainer.links as L
import sys
import matplotlib.pyplot as plt

class LaughNeuralNet(chainer.Chain):
    def __init__(self):
        super(LaughNeuralNet, self).__init__(
            l1=L.Linear(None, 200),
            l2=L.Linear(None, 100),
            l3=L.Linear(None, 2)
        )

    def __call__(self, x):
        h = F.dropout(F.relu(self.l1(x)))
        h = F.dropout(F.relu(self.l2(h)))
        h = self.l3(h)

        return h

class LaughNeuralNet2(chainer.Chain):
    def __init__(self):
        super(LaughNeuralNet2, self).__init__(
            l1=L.Linear(None, 1000),
            b1=L.BatchNormalization(1000),
            l2=L.Linear(None, 1000),
            b2=L.BatchNormalization(1000),
            l3=L.Linear(None, 1000),
            l4=L.Linear(None, 500),
            b3=L.BatchNormalization(500),
            l5=L.Linear(None, 500),
            l6=L.Linear(None, 2)
        )

    def __call__(self, x):
        h = F.dropout(self.b1(F.relu(self.l1(x))))
        h = F.dropout(self.b2(F.relu(self.l2(h))))
        h = F.dropout(F.relu(self.l3(h)))
        h = F.dropout(self.b3(F.relu(self.l4(h))))
        h = F.dropout(F.relu(self.l5(h)))
        h = self.l6(h)

        return h


class LaughNet(chainer.Chain):
    def __init__(self):
        super(LaughNet, self).__init__(
            l1=L.LSTM(None, 200),
            l2=L.Linear(None, 100),
            l3=L.Linear(None, 2)
        )

    def __call__(self, x):
        self.l1.reset_state()
        h = self.l1(x)
        h = F.dropout(self.l2(h))
        h = self.l3(h)

        return h


class LaughNet2(chainer.Chain):
    def __init__(self):
        super(LaughNet2, self).__init__(
            l1=L.Linear(None, 200),
            l2=L.Linear(None, 100),
            l3=L.Linear(None, 2)
        )

    def __call__(self, x):
        h = F.dropout(F.relu(self.l1(x)))
        h = F.dropout(F.relu(self.l2(h)))
        h = self.l3(h)

        return h


# test predicting
def test_predict(path='movie_model'):
    model = LaughNet()

    chainer.serializers.load_npz(path, model)

    with open('confused_data/Raw/v_0.csv', 'r') as f:
        reader = csv.reader(f)
        csv0 = [(np.array(row, dtype=np.float32), 0) for row in reader]

    with open('confused_data/Raw/v_1.csv', 'r') as f:
        reader = csv.reader(f)
        csv1 = [(np.array(row, dtype=np.float32), 1) for row in reader]

    all_ary = csv0 + csv1
    ans_counts = 0
    incorrect = 0
    question_num = len(all_ary)
    plt_ary = []

    for data in all_ary:
        correct = data[1]
        np_data = data[0].reshape(1, 50)
        y = np.argmax(F.softmax(model(np_data)).data)
        if y == correct:
            ans_counts += 1
            plt_ary.append((model(np_data).data, y, 1))
        else:
            incorrect += 1
            plt_ary.append((model(np_data).data, y, 0))

    print('accuracy:', str(ans_counts / question_num))
    print('incorrect rate:', str(incorrect / question_num))

    # ｐｌｔ

    x_ary0 = []
    y_ary0 = []
    x_ary1 = []
    y_ary1 = []

    incorrect_x0 = []
    incorrect_y0 = []

    incorrect_x1 = []
    incorrect_y1 = []

    ax = plt.subplot(111)

    for dt in plt_ary:
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

    np_plt_x1 = np.array(x_ary1, dtype=np.float32)
    np_plt_y1 = np.array(y_ary1, dtype=np.float32)

    np_incorrectx0 = np.array(incorrect_x0, dtype=np.float32)
    np_incorrecty0 = np.array(incorrect_y0, dtype=np.float32)

    np_incorrectx1= np.array(incorrect_x1, dtype=np.float32)
    np_incorrecty1 = np.array(incorrect_y1, dtype=np.float32)

    plt.plot(np_plt_x1, np_plt_y1, 'o', color='r', alpha=0.5)
    plt.plot(np_plt_x0, np_plt_y0, 'o', color='b', alpha=0.5)
    plt.plot(np_incorrectx0, np_incorrecty0, 'o', color='g', alpha=0.5)
    plt.plot(np_incorrectx1, np_incorrecty1, 'o', color='m', alpha=0.5)
    plt.show()

# predict
test_predict()


