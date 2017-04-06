import numpy as np
import csv
import chainer
import chainer.functions as F
import chainer.links as L
import sys

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


# test predicting
def test_predict(path):
    model = None

    if path == 'simple_neural_model':
        model = LaughNeuralNet()
    elif path == 'complex_neural_model':
        model = LaughNeuralNet2()

    if model is None:
        return 'モデルのパラメータがない'

    chainer.serializers.load_npz('params/' + path, model)

    with open('data/Raw/0.csv', 'r') as f:
        reader = csv.reader(f)
        csv0 = [(np.array(row, dtype=np.float32), 0) for row in reader]

    with open('data/Raw/1.csv', 'r') as f:
        reader = csv.reader(f)
        csv1 = [(np.array(row, dtype=np.float32), 1) for row in reader]

    all_ary = csv0 + csv1
    ans_counts = 0
    question_num = len(all_ary)

    for data in all_ary:
        correct = data[1]
        np_data = data[0].reshape(1, 10)
        y = np.argmax(F.softmax(model(np_data)).data)
        if y == correct:
            ans_counts += 1

    print('accuracy:', str(ans_counts / question_num))


# predict
test_predict(sys.argv[1])


