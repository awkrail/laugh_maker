import numpy as np
import csv
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
from chainer.datasets import tuple_dataset
import random


def load_raw_data(number):
    with open('data/Raw/%d.csv' % number, 'r') as f:
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

# row_data_csv_load
np_csv0, np_label0 = load_raw_data(0)
np_csv1, np_label1 = load_raw_data(1)

# all_ary
all_ary = np_csv0 + np_csv1
all_label = np_label0 + np_label1

# after_shuffle_data
all_ary, all_label = shuffle_ary(all_ary, all_label)

# make_chainer_datasets
threshold = np.int32(len(all_ary)/5*4)
train = tuple_dataset.TupleDataset(all_ary[0:threshold], all_label[0:threshold])
test = tuple_dataset.TupleDataset(all_ary[threshold:], all_label[threshold:])



print(train)
