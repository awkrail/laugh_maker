import numpy as np
import chainer
import os
import argparse

import chainer.functions as F
import chainer.links as L
from chainer import cuda
from chainer import Variable
from fix_data import OneDimSensorData

"""
    LSTMを書いて, 訓練するクラス
"""
class LSTM(chainer.Chain):
    def __init__(self):
        super().__init__(
            l1=L.Linear(1, 5),
            l2=L.LSTM(5, 20),
            l3=L.Linear(20, 1),
            l4=L.Linear(1, 2)
        )

    def predictor(self, x):
        # テストデータでの評価用途
        self.l2.reset_state()
        row = x.shape[0]
        col = x.shape[1]
        for i in range(col-1):
            h = self.l1(xp.array(x[:, i].reshape(row, 1), dtype=xp.float32))
            h = self.l2(h)
            h = self.l3(h)
            h = self.l4(h)
        return [0 if data[0] > data[1] else 1 for data in h.data] # WATCH: あとでmapできるようにかきかえる


    def __call__(self, x, t):
        # ひとつのデータごとに誤差を計算する
        self.l2.reset_state()
        row = x.shape[0]
        col = x.shape[1]
        loss = 0
        for i in range(col-1):
            h = self.l1(xp.array(x[:, i].reshape(row, 1), dtype=xp.float32))
            h = self.l2(h)
            h = self.l3(h)
            loss = F.mean_squared_error(h, xp.array(x[:, i+1].reshape(row, 1), dtype=xp.float32))
            h = self.l4(h)
        loss += F.softmax_cross_entropy(h, xp.array(t, dtype=xp.int32))
        # print(loss.data)

        return loss


# gpu
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID')
args = parser.parse_args()

# 学習データとテストデータ
one_dim_sensor = OneDimSensorData()
one_dim_sensor.load_csv()
one_dim_sensor.shuffle()
train, train_label, test, test_label = one_dim_sensor.divide_train_and_test()

# model
model = LSTM()
optimizer = chainer.optimizers.Adam()
optimizer.setup(model)

# for cuda
xp = cuda.cupy if args.gpu >= 0 else np
if args.gpu >= 0:
    cuda.get_device(args.gpu).use()
    model.to_gpu()

# training loop
display = 1000
total_loss = 0
epoch = 30
n_size = len(train)
batch_size = 500
# question_num = len(test)

for i in range(epoch):
    sffindx = np.random.permutation(n_size)
    for j in range(0, n_size, batch_size):
        x = train[sffindx[j:(j+batch_size) if (j+batch_size) < n_size else n_size]]
        y = train_label[sffindx[j:(j+batch_size) if (j+batch_size) < n_size else n_size]]
        loss = model(x, y)
        # print(loss.data)
        model.zerograds()
        loss.backward()
        optimizer.update()
        if j%display == 0:
            print("{} epoch {} number, loss {}".format(i, j, loss.data))
        last_loss = loss.data

    # テストデータでチェック
    answer_num = 0
    for t_d, t_l in zip(test, test_label):
        answers = model.predictor(t_d)
        bool_ans = (answers==t_l)
        for bool in bool_ans:
            if bool:
                answer_num += 1
    print('main/loss {}, accuracy rate {}'.format(last_loss, answer_num/7410))







