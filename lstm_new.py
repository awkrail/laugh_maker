import numpy as np
import chainer
import argparse
import csv

import chainer.functions as F
import chainer.links as L
from chainer import cuda
from chainer import serializers

from fix_data import OneDimSensorData

"""
    LSTMを書いて, 訓練するクラス
"""
class LSTM(chainer.Chain):
    def __init__(self):
        super().__init__(
            l1=L.Linear(1, 5),
            l2=L.LSTM(5, 30),
            l3=L.Linear(30, 1),
            l4=L.Linear(1, 2)
        )

    def predictor(self, x):
        # テストデータでの評価用途
        self.l2.reset_state()
        row = x.shape[0]
        col = x.shape[1]
        for i in range(col):
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
        accum_loss = None
        for i in range(col):
            h = self.l1(xp.array(x[:, i].reshape(row, 1), dtype=xp.float32))
            h = self.l2(h)
            h = self.l3(h)
            if i != col-1:
                loss = F.mean_squared_error(h, xp.array(x[:, i+1].reshape(row, 1), dtype=xp.float32))
                accum_loss = loss if accum_loss is None else accum_loss + loss
            h = self.l4(h)
        accum_loss += F.softmax_cross_entropy(h, xp.array(t, dtype=xp.int32))
        # print(loss.data)

        return accum_loss


# gpu
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID')
args = parser.parse_args()

# 学習データとテストデータ
one_dim_sensor = OneDimSensorData()
one_dim_sensor.load_csv()
one_dim_sensor.shuffle()
train, train_label, test, test_label, one_label_counts = one_dim_sensor.divide_train_and_test()

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
epoch = 500
n_size = len(train)
batch_size = 1000
# question_num = len(test)

loss_plt_ary = []
accuracy_plt_ary = []
recall_plt_ary = []


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
    recall = 0
    for t_d, t_l in zip(test, test_label):
        answers = model.predictor(t_d)
        bool_ans = (answers==t_l)
        for bool in bool_ans:
            if bool:
                answer_num += 1
        for answer, tl in zip(answers, t_l):
            if answer == tl and answer:
                recall += 1
    print('main/loss {}, accuracy rate {}, recall rate {}, one_counts {}'.format(last_loss, answer_num/7410, recall/one_label_counts, one_label_counts))
    loss_plt_ary.append(last_loss)
    accuracy_plt_ary.append(answer_num/7410)
    recall_plt_ary.append(recall/one_label_counts)


serializers.save_npz("fixed_lstm_model_2", model)

def write_csv(path, ary):
    with open('results/lstm_fixed_result/%s.csv' % path, 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(ary)

write_csv('loss', loss_plt_ary)
write_csv('accuracy', accuracy_plt_ary)
write_csv('recall', recall_plt_ary)

# csvにデータを書き込み
with open('results/lstm_fixed_result/loss.csv', 'w') as f:
    writer = csv.writer(f, lineterminator='\n')
    writer





