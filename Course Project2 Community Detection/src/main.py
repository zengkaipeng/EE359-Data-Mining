import csv
import pandas
import Louvinpy
import json


def Generate_belong(partition):
    belong = {}
    for k, v in partition.items():
        for node in v:
            belong[node] = k
    return belong


def Extend_Labels(Gt, edges_of_nodes):
    Score = {}
    for node, mark in Gt:
        node, mark = int(node), int(mark)
        Score[node] = {mark: 1}
        for neighbor in edges_of_nodes[node]:
            neighbor = int(neighbor)
            if neighbor == node:
                continue
            else:
                temp = Score.get(neighbor, {})
                temp[mark] = temp.get(mark, 0) + 1
                Score[neighbor] = temp
    for k, v in Score.items():
        tot = sum(v.values())
        for Marker in v.keys():
            Score[k][Marker] /= tot
    return Score


def Vote(Gt, partition, Score_Board):
    Voter, Unlabeled, LabelCnt = {}, [], {}
    for k, v in partition.items():
        Count = {}
        for Node, Mark in Gt:
            if Node in v:
                Count[Mark] = Count.get(Mark, 0) + 1
        if len(Count) == 0:
            Voter[k] = 0
            Unlabeled.append(k)
        else:
            ansid, anscnt = 0, -1
            issame, cands = False, []
            for Mk, Cn in Count.items():
                if Cn > anscnt:
                    anscnt, ansid = Cn, Mk
                    issame, cands = False, [Mk]
                elif Cn == anscnt:
                    issame = True
                    cands.append(Mk)
            Voter[k] = ansid
            LabelCnt[ansid] = LabelCnt.get(ansid, 0) + anscnt
            """
            if issame:
                # print("Here", len(v), ansid)
                Count2 = {}
                for Node, Scores in Score_Board.items():
                    for mark, sco in Scores.items():
                        Count2[mark] = Count2.get(mark, 0) + sco
                # print(cands, anscnt)
                Tans = cands[0]
                for tx in cands:
                    if Count2.get(tx, 0) > Count2.get(Tans, 0):
                        Tans = tx
                Voter[k] = Tans
                # print("There", Tans)
            """
    Minimal_Cnt = 0
    for k, v in LabelCnt.items():
        if v < LabelCnt.get(Minimal_Cnt, 0):
            Minimal_Cnt = k
    for x in Unlabeled:
        # Voter[x] = 0
        Voter[x] = Minimal_Cnt
    # print(LabelCnt)
    return Voter


if __name__ == '__main__':
    edges = pandas.read_csv('../data/edges.csv').values
    Nodes, edges_of_nodes = set(), {}
    for x, y in edges:
        Nodes.add(x)
        Nodes.add(y)
        if x not in edges_of_nodes:
            edges_of_nodes[x] = []
        if y not in edges_of_nodes:
            edges_of_nodes[y] = []
        edges_of_nodes[x].append(y)
        edges_of_nodes[y].append(x)

    # print(len(Nodes))
    Gt = pandas.read_csv('../data/ground_truth.csv').values
    Solver = Louvinpy.Louvinpy(edges, Nodes)
    partition, best_q = Solver.Solve()
    Belong = Generate_belong(partition)
    Score_Board = Extend_Labels(Gt, edges_of_nodes)
    # print(Score_Board)
    # with open('../data/Extend_Labels.json', 'w') as Fout:
    #     Fout.write(json.dumps(Score_Board, indent=4))

    Marker = Vote(Gt, partition, Score_Board)
    Ans = []
    for node in Nodes:
        Ans.append({
            'id': node,
            'category': Marker[Belong[node]]
        })

    with open('../data/answer.csv', 'w', newline='') as Fout:
        f_csv = csv.DictWriter(Fout, ['id', 'category'])
        f_csv.writeheader()
        f_csv.writerows(Ans)
