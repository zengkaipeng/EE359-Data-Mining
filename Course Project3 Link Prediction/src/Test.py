import Node2Vec
import pandas as pds
import csv
import time
from random import shuffle
from random import randint
import json
from sklearn.metrics import roc_auc_score


def NegSam_e(Num, Nodelist, Pset):
    ans = []
    while len(ans) < Num:
        x = randint(0, len(Nodelist) - 1)
        y = randint(0, len(Nodelist) - 1)
        if x != y and (Nodelist[x], Nodelist[y]) not in Pset:
            ans.append([x, y])
    return ans


if __name__ == '__main__':
    Input_file = '../data/course3_edge.csv'
    Edges = pds.read_csv(Input_file).values
    shuffle(Edges)

    Test_Edges = Edges[:4500]
    Used_Edges = Edges[4500:]
    Positive_Set = set()
    for x, y in Used_Edges:
        Positive_Set.add((x, y))
        Positive_Set.add((y, x))
    Test_Neg = NegSam_e(4500, list(range(16863)), Positive_Set)

    Nodes = set(range(16863))

    Model = Node2Vec.Node2Vec(
        edges=Used_Edges, nodes=Nodes, walk_leng=25, dims=128,
        nevsam=30, p=0.5, q=2, verbose=True
    )

    Model.fit(epoch=60)

    Positive_Prob = Model.Predict(Test_Edges)
    Neg_Prob = Model.Predict(Test_Neg)

    with open('../data/pos_pred.jsonl', 'w') as Fout:
        for i, prob in enumerate(Positive_Prob):
            Fout.write('{}\n'.format(json.dumps(
                [Test_Edges[i].tolist(), prob]
            )))

    with open('../data/neg_pred.jsonl', 'w') as Fout:
        for i, prob in enumerate(Neg_Prob):
            Fout.write('{}\n'.format(json.dumps(
                [Test_Neg[i], prob]
            )))

    print('Score: ', roc_auc_score(
        [1] * 4500 + [0] * 4500,
        Positive_Prob + Neg_Prob
    ))
