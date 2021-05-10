import csv
import numpy as np
from tqdm import tqdm
import pandas as pds
from random import randint
import argparse

ArgPs = argparse.ArgumentParser()
ArgPs.add_argument(
    '-c', '--category', default=5, type=int,
    help='the number of categories, default is 5'
)
ArgPs.add_argument(
    '-i', '--input', default='../data/course1.csv',
    help='the path of input file, default is ../data/course1.csv'
)
ArgPs.add_argument(
    '-o', '--output', default='../data/Answer.csv',
    help='the path of output file, default is ../data/Answer.csv'
)
ArgPs.add_argument(
    '-iter', '--iteration', default=200, type=int,
    help='Max number of iteration, default is 200'
)
args = vars(ArgPs.parse_args())

Info = [
    'number of categories: {}'.format(args['category']),
    'number of iterations: {}'.format(args['iteration']),
    'input file: {}'.format(args['input']),
    'output file: {}'.format(args['output'])
]

InfoLen = max(len(x) for x in Info)
print('*' * (InfoLen + 4))
for info in Info:
    print('* {}{} *'.format(info, ' ' * (InfoLen - len(info))))
print('*' * (InfoLen + 4))


def Select_Inits(Data, K):
    num, ans = Data.shape[0], []
    ans.append(randint(0, num - 1))
    for iterid in range(1, K):
        difference_square = [(Data - Data[x]) * (Data - Data[x]) for x in ans]
        Distances = np.array(difference_square).sum(axis=2)
        Mindistance = Distances.min(axis=0)
        ans.append(Mindistance.argmax())
        # print(ans[-1], Mindis[ans[-1]])
    return ans


file_name = args['input']
Kind = args['category']

print('[INFO] Reading from input file')

Data_row = pds.read_csv(file_name).values
Data, Idx2Pid, Cnt = [], {}, 0
for i in Data_row:
    Idx2Pid[Cnt] = int(i[0])
    Data.append(i[1:])
    Cnt += 1
Data = np.array(Data, dtype=np.float64)

InitPoints = Select_Inits(Data, Kind)

LastBelong = np.zeros(Cnt)
Cores = np.array([Data[x] for x in InitPoints])


max_iter = args['iteration']

print("[INFO] Iteration Begin")

for iteridx in range(max_iter):
    if iteridx % 10 == 0:
        print("{} iter of {} in total".format(iteridx + 1, max_iter))

    difference_square = [(Data - x) * (Data - x) for x in Cores]
    Distances = np.array(difference_square).sum(axis=2)
    # print(Distances.shape)
    Belong = Distances.argmin(axis=0)
    Same, Flag = (Belong == LastBelong), True

    for x in range(Kind):
        Totsize = len(np.where(Belong == x)[0])
        Cores[x] = np.sum(Data[Belong == x], axis=0) / float(Totsize)

    for x in Same:
        Flag &= x
    if Flag:
        break
    else:
        LastBelong = Belong


print("[INFO] Iteration End")

Radius = [-1] * Kind
for idx, val in enumerate(Belong):
    Dist = ((Data[idx] - Cores[val]) * (Data[idx] - Cores[val])).sum()
    Radius[val] = max(Radius[val], Dist)

category_list = [x for x in range(Kind)]
category_list.sort(key=lambda x: Radius[x])
reversed_category = {}
for idx, val in enumerate(category_list):
    reversed_category[val] = idx
# print(Radius, reversed_category)

# print(Belong)
Ans = []
for idx, bel in enumerate(Belong):
    Ans.append({
        'id': Idx2Pid[idx],
        'category': reversed_category[bel]
    })

with open(args['output'], 'w', newline='') as Fout:
    f_csv = csv.DictWriter(Fout, ['id', 'category'])
    f_csv.writeheader()
    f_csv.writerows(Ans)
