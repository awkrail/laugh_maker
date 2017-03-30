import numpy as np
import csv
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
from chainer.datasets import tuple_dataset

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

# test predicting
def test_predict():
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


# model set
model = LaughNeuralNet()
chainer.serializers.load_npz("neural_model", model)
# predict
"""
pdt = np.array([0.38369206, 0.386085559, 0.412592659, 0.436813432, 0.458962222, 0.481111012, 0.519335536, 0.554166295, 0.588639814, 0.61436099
], dtype=np.float32)
pdt = pdt.reshape(1, 10)
y = model(pdt)
print(F.sigmoid(y).data)
"""
test_predict()


