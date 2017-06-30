import numpy as np
import csv
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
from chainer.datasets import tuple_dataset
import random
import os


def load_raw_data(number):
    with open('confused_data/five_sec/%d.csv' % number, 'r') as f:
        reader = csv.reader(f)
        np_data = [np.array(row, dtype=np.float32) for row in reader]

    label_data = [np.int32(number) for _ in np_data]

    return np_data, label_data


def shuffle_ary(np_ary, label_ary):
    before_ary = [(np_data, label) for np_data, label in zip(np_ary, label_ary)]
    random.shuffle(before_ary)
    af_np_ary = [data[0] for data in before_ary]
    af_label_ary = [data[1] for data in before_ary]

    return af_np_ary, af_label_ary


class DailyDataLoader(object):
    def __init__(self):
        self.tuple_ary = []

    def load_csv(self):
        directories = os.listdir('daily_data/mean')
        for directory in directories:
            with open('daily_data/mean/' + directory, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    self.tuple_ary.append((np.array(row[:-1],dtype=np.float32), np.array(row[-1], dtype=np.int32)))

    def shuffle_ary(self):
        random.shuffle(self.tuple_ary)
        af_data_ary = [tuple_data[0] for tuple_data in self.tuple_ary]
        af_label_ary = [tuple_data[1] for tuple_data in self.tuple_ary]

        return af_data_ary, af_label_ary


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

# model define
model = LaughNet()
classify_model = L.Classifier(model)
optimizer = chainer.optimizers.Adam()
optimizer.setup(classify_model)

# row_data_csv_load
# np_csv0, np_label0 = load_raw_data(0)
# np_csv1, np_label1 = load_raw_data(1)

# print(len(np_csv0[0]))

# all_ary
# all_ary = np_csv0 + np_csv1
# all_label = np_label0 + np_label1

# after_shuffle_data
# all_ary, all_label = shuffle_ary(all_ary, all_label)

daily_data = DailyDataLoader()
daily_data.load_csv()
all_ary, all_label = daily_data.shuffle_ary()

# import ipdb; ipdb.set_trace()

# make_chainer_datasets and iterator
threshold = np.int32(len(all_ary)/5*4)
train = tuple_dataset.TupleDataset(all_ary[0:threshold], all_label[0:threshold])
test = tuple_dataset.TupleDataset(all_ary[threshold:], all_label[threshold:])

train_iter = chainer.iterators.SerialIterator(train, 100, shuffle=True)
test_iter = chainer.iterators.SerialIterator(test, 100, repeat=False, shuffle=False)

updater = training.StandardUpdater(train_iter, optimizer, device=0)
trainer = training.Trainer(updater, (40, 'epoch'), out='result3')
trainer.extend(extensions.Evaluator(test_iter, classify_model, device=0))
trainer.extend(extensions.LogReport())
trainer.extend(extensions.PlotReport(y_keys='main/loss', file_name='main_loss.png'))
trainer.extend(extensions.PlotReport(y_keys='main/accuracy', file_name='main_acc.png'))
trainer.extend(extensions.PlotReport(y_keys='validation/main/loss', file_name='overfitting.png'))
trainer.extend(extensions.PlotReport(y_keys='validation/main/accuracy', file_name='validation_acc.png'))

trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'validation/main/loss', 'main/accuracy', 'validation/main/accuracy']))

trainer.run()

chainer.serializers.save_npz("movie_model_simple", model)

















