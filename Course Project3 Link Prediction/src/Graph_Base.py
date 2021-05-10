import json
import warnings
import numpy as np
from tqdm import tqdm


class Graph:
    def _add_edge(self, x, y):
        if x not in self.edges:
            self.edges[x] = {}
        self.in_degree[y] = self.in_degree.get(y, 0) + \
            (1 if self.repeat or y not in self.edges[x] else 0)
        self.out_degree[x] = self.out_degree.get(x, 0) + \
            (1 if self.repeat or y not in self.edges[x] else 0)

        self.edges[x][y] = 1 + \
            (self.edges[x].get(y, 0) if self.repeat else 0)

    def __init__(self, edges, nodes=None, repeat=True,  directed=False):
        _nodes = set()
        self.edges = {}
        self.repeat = repeat
        self.directed = directed
        self.in_degree = {}
        self.out_degree = {}
        _nodes = set()
        for x, y in edges:
            self._add_edge(x, y)
            if not directed:
                self._add_edge(y, x)
            _nodes.add(x)
            _nodes.add(y)

        self.nodes = _nodes if nodes is None else set(nodes)

    def get_nodes(self):
        return self.nodes.copy()

    def get_edges(self):
        return self.edges.copy()

    def get_edges_list(self):
        ans = []
        for x, y in self.edges.items():
            for k, v in y.items():
                for p in range(v):
                    ans.append((x, k))
        return ans

    def get_in_degree(self):
        return self.in_degree.copy()

    def get_out_degree(self):
        return self.out_degree.copy()

    def get_degree(self):
        ans = {}
        for k, v in self.in_degree.items():
            ans[k] = ans.get(k, 0) + v
        for k, v in self.out_degree.items():
            ans[k] = ans.get(k, 0) + v
        if not self.directed:
            for k in ans:
                ans[k] //= 2
        return ans

    def get_node_list(self):
        return list(self.nodes)

    def get_degree_point(self, point):
        ans = self.in_degree.get(point, 0) + self.out_degree.get(point, 0)
        return ans if self.directed else ans // 2


class WalkGraph(Graph):
    def __init__(
        self, edges, p=1, q=1,
        nodes=None, repeat=True, directed=False
    ):
        super(WalkGraph, self).__init__(
            edges=edges, nodes=nodes,
            repeat=repeat, directed=directed
        )
        self.q, self.invq = q, 1 / q
        self.p, self.invp = p, 1 / p

    def get_val(self, v, u, w):
        if v == w:
            return self.invp
        elif v in self.edges.get(w, {}):
            return 1
        else:
            return self.invq

    def dfs(self, now, last, res, ans):
        if res == 0:
            return
        Neighbors = self.edges[now]
        Keys = list(Neighbors.keys())
        Distributions = np.array(
            [Neighbors[v] * self.get_val(v, now, last) for v in Keys],
            dtype=np.float64
        )
        Distributions = Distributions / Distributions.sum()
        Test = np.random.multinomial(1, Distributions)
        ans.append(Keys[Test.argmax()])
        self.dfs(ans[-1], now, res - 1, ans)

    def Walk_Single(self, point, length):
        if length < 0:
            raise ValueError('walk length should be positive')
        elif self.directed:
            Msg = 'Walk shouldn\'t be performed on directed graph'
            raise AttributeError(Msg)
        else:
            if self.get_degree_point(point) == 0:
                return [point for x in range(length + 1)]
            else:
                ans = [point]
                self.dfs(point, None, length, ans)
                return ans

    def Walks(self, length, verbose=False):
        if verbose:
            print('[INFO] Random Walking..')

        ans = []
        Iters = tqdm(self.nodes) if verbose else self.nodes
        for i in Iters:
            ans.append(self.Walk_Single(i, length))

        if verbose:
            print('[INFO] Random Walking Done')

        return ans


if __name__ == '__main__':
    V = WalkGraph(
        [
            [1, 2], [1, 2], [2, 3], [4, 5],
            [3, 4], [5, 7], [6, 8], [9, 1],
            [9, 2], [5, 9], [8, 3], [4, 9],
            [6, 2], [7, 1]
        ],
        repeat=True, p=0.5, q=2,
        nodes=[1, 2, 3, 4, 5, 6, 7, 9]
    )
    print(V.get_edges())
    print(V.get_edges_list())
    print(V.get_degree())
    print(V.get_in_degree())
    print(V.get_out_degree())

    print('\n\n')
    print(V.Walks(20, True))
