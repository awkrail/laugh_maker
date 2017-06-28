import chainer
import chainer.links as L
import chainer.functions as F
from chainer.datasets import tuple_dataset
from fix_data import MatrixSensorData
from chainer import training
from chainer.training import extensions


class ConvolutionFilterNet(chainer.Chain):
    def __init__(self):
        super(ConvolutionFilterNet, self).__init__(
            l1=L.Convolution2D(50, 100, 2, 1, 0),
            l2=L.Convolution2D(100, 150, 2, 1, 0),
            l3=L.Linear(None, 200),
            l4=L.Linear(None, 2)
        )

    def __call__(self, x):
        h = F.max_pooling_2d(F.relu(self.l1(x)), 0, 1)
        h = F.max_pooling_2d(F.relu(self.l2(h)), 0, 1)
        h = F.dropout(F.relu(self.l3(h)))
        h = self.l4(h)

        return h


class ConvolutionFilterNet2(chainer.Chain):
    def __init__(self):
        super(ConvolutionFilterNet2, self).__init__(
            l1=L.Convolution2D(None, 100, 2, 1, 0),
            l2=L.Convolution2D(None, 150, 2, 1, 0),
            l3=L.Linear(None, 200),
            l4=L.Linear(None, 2)
        )

    def __call__(self, x):
        h = F.max_pooling_2d(F.relu(self.l1(x)), 2)
        h = F.max_pooling_2d(F.relu(self.l2(h)), 2)
        h = F.dropout(F.relu(self.l3(h)))
        h = self.l4(h)

        return h


# setup model
model = ConvolutionFilterNet2()
classifier_model = L.Classifier(model)
optimizer = chainer.optimizers.Adam()
optimizer.setup(classifier_model)

# data
all_data = MatrixSensorData()
all_data.load_csv()
train, test = all_data.divide_train_and_validation()

# training
train_iter = chainer.iterators.SerialIterator(train, 100, shuffle=True)
test_iter = chainer.iterators.SerialIterator(test, 100, repeat=False, shuffle=False)

updater = training.StandardUpdater(train_iter, optimizer, device=0)
trainer = training.Trainer(updater, (50, 'epoch'), out='result3')
trainer.extend(extensions.Evaluator(test_iter, classifier_model, device=0))
trainer.extend(extensions.LogReport())
trainer.extend(extensions.PlotReport(y_keys='main/loss', file_name='main_loss.png'))
trainer.extend(extensions.PlotReport(y_keys='main/accuracy', file_name='main_acc.png'))
trainer.extend(extensions.PlotReport(y_keys='validation/main/loss', file_name='overfitting.png'))
trainer.extend(extensions.PlotReport(y_keys='validation/main/accuracy', file_name='validation_acc.png'))

trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'validation/main/loss', 'main/accuracy', 'validation/main/accuracy']))

trainer.run()

chainer.serializers.save_npz("CNN_model", model)

