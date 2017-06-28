import numpy as np
import csv
import chainer
import chainer.functions as F
import chainer.links as L
import sys
import matplotlib.pyplot as plt

# 利用するニューラルネットワーク
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

# testscript
if __name__ == '__main__':
    model = LaughNet()
    chainer.serializers.load_npz('movie_model', model)
    all_ary = []

    with open('validation.csv', 'r') as f:
        for row in csv.reader(f):
            all_ary.append((np.array(row[:-1], dtype=np.float32), int(row[-1])))

    ans_counts = 0
    incorrect_counts = 0
    question_num = len(all_ary)
    plt_ary = []
    answer_row = []

    for i, data in enumerate(all_ary):
        val_np = data[0].reshape(1, 50)
        answer = data[1]
        y = np.argmax(F.softmax(model(val_np)).data)
        answer_row.append([i, y])
        if y == answer:
            ans_counts += 1
            plt_ary.append((model(val_np).data, y, 1))
        else:
            incorrect_counts += 1
            plt_ary.append((model(val_np).data, y, 0))

    print('accuracy_rate: ', str(ans_counts/question_num))
    print('incorrect rate:', str(incorrect_counts / question_num))

    with open('answer_row', 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(answer_row)

    x_ary0 = []
    y_ary0 = []
    x_ary1 = []
    y_ary1 = []

    incorrect_x0 = []
    incorrect_y0 = []

    incorrect_x1 = []
    incorrect_y1 = []

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

    print('1だとおもったら->0: ', len(incorrect_x0))
    print('0だとおもったら->1: ', len(incorrect_x1))
    print('1だとおもったら->1: ', len(x_ary1))
    print('0だとおもったら->0: ', len(x_ary0))

    np_plt_x1 = np.array(x_ary1, dtype=np.float32)
    np_plt_y1 = np.array(y_ary1, dtype=np.float32)

    np_incorrectx0 = np.array(incorrect_x0, dtype=np.float32)
    np_incorrecty0 = np.array(incorrect_y0, dtype=np.float32)

    np_incorrectx1= np.array(incorrect_x1, dtype=np.float32)
    np_incorrecty1 = np.array(incorrect_y1, dtype=np.float32)

    plt.plot(np_plt_x1, np_plt_y1, 'o', color='r', alpha=0.5)
    plt.plot(np_plt_x0, np_plt_y0, 'o', color='b', alpha=0.5)
    # plt.plot(np_incorrectx0, np_incorrecty0, 'o', color='g', alpha=0.5)
    # plt.plot(np_incorrectx1, np_incorrecty1, 'o', color='m', alpha=0.5)
    plt.show()

