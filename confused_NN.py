import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
import random
from chainer import training
from chainer.training import extensions
from chainer.datasets import tuple_dataset


# データを読み込み, 管理するクラス
class ProcessData(object):
    def __init__(self, fpath, number):
        self.fpath = fpath
        self.label = number

    def load_data(self):
        np_data = np.loadtxt(self.fpath, delimiter=',', dtype=np.float32)
        if self.label == 0:
            np_label = np.zeros((np_data.shape[0], 1))
        else:
            np_label = np.ones((np_data.shape[0], 1))

        for data, label in zip(np_data, np_label):
            yield (data, label)

    @staticmethod
    def shuffle(np0, np1):
        all_ary = np0 + np1
        random.shuffle(all_ary)
        for data, label in all_ary:
            yield (data, label)


# ニューラルネットワーク
class LaughNet(chainer.Chain):
    def __init__(self):
        super(LaughNet, self).__init__(
            l1=L.LSTM(None, 20),
            l2=L.Linear(None, 100),
            l3=L.Linear(None, 2)
        )

    def __call__(self, x):
        self.l1.reset_state()
        h = self.l1(x)
        h = F.dropout(self.l2(h))
        h = self.l3(h)

        return h


def main():
    zero_data = ProcessData('confused_data/Raw/fixed_0.csv', 0)
    one_data = ProcessData('confused_data/Raw/1.csv', 1)

    zero_gen = zero_data.load_data()
    one_gen = one_data.load_data()

    all_data = ProcessData.shuffle(list(zero_gen), list(one_gen))
    all_ary = list(all_data)


    # make_chainer_datasets and iterator
    threshold = np.int32(len(all_ary) / 5 * 4)
    train = tuple_dataset.TupleDataset(all_ary[0:threshold][0], all_ary[0:threshold][1])
    test = tuple_dataset.TupleDataset(all_ary[threshold:][0], all_ary[threshold:][1])

    # model define
    model = LaughNet()
    classify_model = L.Classifier(model)
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(classify_model)
    train_iter = chainer.iterators.SerialIterator(train, 100, shuffle=True)
    test_iter = chainer.iterators.SerialIterator(test, 100, repeat=False, shuffle=False)

    updater = training.StandardUpdater(train_iter, optimizer, device=-1)
    trainer = training.Trainer(updater, (100, 'epoch'), out='result')
    trainer.extend(extensions.Evaluator(test_iter, classify_model, device=-1))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PlotReport(y_keys='main/loss', file_name='main_loss.png'))
    trainer.extend(extensions.PlotReport(y_keys='main/accuracy', file_name='main_acc.png'))
    trainer.extend(extensions.PlotReport(y_keys='validation/main/loss', file_name='overfitting.png'))
    trainer.extend(extensions.PlotReport(y_keys='validation/main/accuracy', file_name='validation_acc.png'))

    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss', 'main/accuracy', 'validation/main/accuracy']))

    trainer.run()

if __name__ == '__main__':
    main()








