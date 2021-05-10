import NegativeSampler
import Graph_Base
from random import shuffle
from tqdm import tqdm
import numpy as np
import math
from collections import Counter
import time
from random import randint


class StepLrer:
    def __init__(self, lr=1e-3, gamma=0.1, step=5, minimal=5e-7):
        self.lr = lr
        self.gamma = gamma
        self.step = step
        self.curr = 0
        self.minimal = minimal

    def Get_lr(self):
        ans = self.lr
        self.curr += 1
        if self.curr == self.step:
            self.curr, self.lr = 0, self.lr * self.gamma
            self.lr = max(self.lr, self.minimal)
        return ans


class Node2Vec:

    def Pre_Process(self, edges, nodes):
        self.node2mask, self.mask2node = {}, {}
        if nodes == None:
            _nodes = set()
            for x, y in edges:
                _nodes.add(x)
                _nodes.add(y)
        else:
            _nodes = nodes.copy()
        curr = 0
        for x in _nodes:
            self.node2mask[x] = curr
            self.mask2node[curr] = x
            curr += 1
        _edges = []
        for x, y in edges:
            _edges.append((
                self.node2mask[x], self.node2mask[y]
            ))
        return _edges, set(range(curr))

    def __init__(
        self, edges, dims, walk_leng=20, nodes=None, nevsam=20,
        p=1, q=1, verbose=False,
    ):
        edges, nodes = self.Pre_Process(edges, nodes)

        self.G = Graph_Base.WalkGraph(
            edges, nodes=nodes, p=p, q=q
        )
        self.is_trainned = False
        self.Nodes = self.G.get_node_list()
        self.dim = dims
        self.verbose = verbose
        self.walk_leng = walk_leng
        self.nevsam = nevsam
        self.Datasize = len(self.Nodes)
        self.NegSampler = NegativeSampler.NegativeSampler(
            self.G.get_degree(), verbose=self.verbose
        )
        self.walktime = 20
        self._Generate_Neighbors()

    def _Generate_Neighbors(self):
        if self.verbose:
            print('[INFO] Generating Neighbors...')

        self.WalkPools = []
        for i in range(self.walktime):
            ans = self.G.Walks(self.walk_leng, self.verbose)
            self.WalkPools.append(ans)
        self.WalkPools = np.concatenate(self.WalkPools).tolist()
        shuffle(self.WalkPools)

        if self.verbose:
            print('[INFO] Done')

    def _sigmoid(self, arr):

        try:
            ans = 1 / (1 + math.exp(-arr))
        except OverflowError as e:
            ans = 1e-5
        return ans
        """
        ans = 1 / (1 + math.exp(-arr))
        return ans
        """

    def fit(self, epoch=100):
        if self.verbose:
            print('[INFO] Start Training')
        StepLr = StepLrer(lr=1e-2, gamma=0.1, step=15)
        self.Embeddings = np.random.normal(
            loc=0, scale=0.001,
            size=(self.Datasize, self.dim)
        )

        for i in range(epoch):
            Lr = StepLr.Get_lr()
            epoch_start_time = time.time()
            totlen = self.Datasize * self.walktime
            pos = randint(0, (self.walktime - 3) * self.Datasize)
            grad = np.zeros(
                shape=self.Embeddings.shape,
                dtype=np.float64
            )
            for Walk in self.WalkPools[pos: pos + 3 * self.Datasize]:
                if Walk[0] == Walk[1]:
                    continue

                u = Walk[0]
                for v in Walk[1:]:
                    one_sg = self._sigmoid(
                        -1 * self.Embeddings[u].dot(self.Embeddings[v])
                    )
                    grad[u] -= one_sg * self.Embeddings[v]
                    grad[v] -= one_sg * self.Embeddings[u]

                NegSamples = self.NegSampler.Sample(self.nevsam)

                for sam in NegSamples:
                    one_sg = self._sigmoid(
                        -1 * self.Embeddings[u].dot(self.Embeddings[sam])
                    )
                    grad[u] += one_sg * self.Embeddings[sam]
                    grad[sam] += one_sg * self.Embeddings[u]

            self.Embeddings -= Lr * grad

            if self.verbose:
                print('[INFO] epoch = {} / {} Lr = {} {:.4f}s Used'.format(
                    i + 1, epoch, Lr, time.time() - epoch_start_time
                ))
                print('-----------------------------------------')
        self.is_trainned = True
        if self.verbose:
            print('[INFO] Done')

    def Predict(self, Pairs):
        if not self.is_trainned:
            raise AttributeError('Not Fit Yet')
        ans = []
        for x, y in Pairs:
            mask1 = self.node2mask.get(x, -1)
            mask2 = self.node2mask.get(y, -1)
            Encoded1 = self.Get_Encoded(mask1)
            Encoded2 = self.Get_Encoded(mask2)
            Norm1 = Encoded1.dot(Encoded1) ** 0.5
            Norm2 = Encoded2.dot(Encoded2) ** 0.5
            Consine_Sim = Encoded1.dot(Encoded2) / (Norm1 * Norm2)
            ans.append(Consine_Sim)
        return ans

    def Get_Encoded(self, mask):
        if mask == -1:
            return np.random.normal(
                loc=0, scale=1 / (self.dim ** 0.5),
                size=[self.dim]
            )
        else:
            return self.Embeddings[mask]

    def Get_Embedding_numpy_and_mask(self):
        if not self.is_trainned:
            raise AttributeError('Not Fit Yet')
        return np.array(self.Embeddings), self.node2mask.copy()

    def Get_Embeddings(self):
        if not self.is_trainned:
            raise AttributeError('Not Fit Yet')
        Ans = {}
        for mask, node in self.mask2node.items():
            Ans[node] = self.Embeddings[mask]
        return Ans
