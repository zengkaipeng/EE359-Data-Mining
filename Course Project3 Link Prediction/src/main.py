import Node2Vec
import pandas as pds
import csv
import time
import pickle

if __name__ == '__main__':
    Input_file = '../data/course3_edge.csv'
    Edges = pds.read_csv(Input_file).values
    Nodes = set(range(16863))

    Model = Node2Vec.Node2Vec(
        edges=Edges, nodes=Nodes, walk_leng=25, dims=128,
        nevsam=30, p=0.5, q=2, verbose=True
    )

    starter = time.time()
    Model.fit(epoch=60)
    with open('../data/Time.txt', 'w') as Fout:
        Fout.write('{} seconds'.format(time.time() - starter))
    with open('../data/Embeddings.pkl', 'wb') as Fout:
        pickle.dump(Model.Get_Embeddings(), Fout)

    Test_file = '../data/course3_test.csv'
    Test_Edges = pds.read_csv(Test_file).values
    Test_Edges = Test_Edges[:, 1:]
    Answer = Model.Predict(Test_Edges)
    Anss = []
    for idx, Ans in enumerate(Answer):
        Anss.append({
            'id': idx,
            'label': round(Ans, 4)
        })

    with open('../data/submission.csv', 'w', newline='') as Fout:
        f_csv = csv.DictWriter(Fout, ['id', 'label'])
        f_csv.writeheader()
        f_csv.writerows(Anss)
