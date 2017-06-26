import csv

def apart_train_valid(number, ary, valid_ary):
    with open('confused_data/Raw/%d.csv' % number, 'r') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if i < 16000:
                ary.append([data for i, data in enumerate(row) if i != 50])
            else:
                valid_ary.append([data for i,data in enumerate(row) if i != 50]) 
    return ary, valid_ary

def write_train_valid(number, ary, valid_ary):
    with open('confused_data/Raw/fixed_%d.csv' % number, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(ary) 
    with open('confused_data/Raw/fixed_valid_%d.csv' % number, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(valid_ary)


ary0 = []
valid_0_ary = []
ary1 = []
valid_1_ary = []

ary0, valid_0_ary = apart_train_valid(0, ary0, valid_0_ary)
ary1, valid_1_ary = apart_train_valid(1, ary1, valid_1_ary)

print('length_ary0', len(ary0), len(ary0[0]), len(valid_0_ary), len(valid_0_ary[0]))
print('length_ary1', len(ary1), len(ary1[0]), len(valid_1_ary), len(valid_1_ary[0]))

write_train_valid(0, ary0, valid_0_ary)
write_train_valid(1, ary1, valid_1_ary)
