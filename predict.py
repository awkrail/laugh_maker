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

# model set
model = LaughNeuralNet()
chainer.serializers.load_npz("neural_model", model)
# predict
pdt = np.array([0.38369206, 0.386085559, 0.412592659, 0.436813432, 0.458962222, 0.481111012, 0.519335536, 0.554166295, 0.588639814, 0.61436099
], dtype=np.float32)
pdt = pdt.reshape(1, 10)
y = model(pdt)
print(F.sigmoid(y).data)