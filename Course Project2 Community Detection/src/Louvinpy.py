from tqdm import tqdm


class Partition:
    def __init__(self, nodes, tot=0):
        self.nodes = set(nodes)
        self.s_in = 0
        self.s_tot = tot


class Louvinpy(object):
    def __init__(self, edges, nodes):
        super(Louvinpy, self).__init__()
        self.edges = edges
        self.Nodes = set(nodes)
        self.m = 0
        self.ki = {}
        self.edges_of_nodes = {}
        self.degrees = {}
        self.w = {}
        self.belong = {}
        for x in nodes:
            self.edges_of_nodes[x] = []
            self.belong[x] = x
        for x, y in edges:
            self.m += 1
            self.ki[x] = self.ki.get(x, 0) + 1
            self.ki[y] = self.ki.get(y, 0) + 1
            self.edges_of_nodes[x].append((x, y, 1))
            if x != y:
                self.edges_of_nodes[y].append((y, x, 1))
            self.degrees[x] = self.degrees.get(x, 0) + 1
            self.degrees[y] = self.degrees.get(y, 0) + 1

        self.actual_partiton = {}

    def Modularity(self, spartition):
        ans, m2 = 0, 2 * self.m
        for xpart in spartition.values():
            ans += xpart.s_in / m2 + (xpart.s_tot / m2) ** 2
        return ans

    def Modularity_Gain(self, node, clu, ki_in):
        return ki_in * 2 - clu.s_tot * self.ki[node] / self.m

    def Neighbors(self, node):
        for x, y, val in self.edges_of_nodes[node]:
            if x == y:
                continue
            else:
                yield y

    def init_partition(self, nodes, edges):
        self.belong, partition = {}, {}
        for node in nodes:
            partition[node] = Partition([node], self.ki.get(node, 0))
            self.belong[node] = node
        for x, y, val in edges:
            if x == y:
                partition[x].s_in += val
                partition[y].s_in += val
        return partition

    def first_phase(self, nodes, edges):
        # print("fstiter !!")
        best_partition = self.init_partition(nodes, edges)
        # print("OK")
        # idx = 1
        while True:
            Improved = False
            for node in nodes:
                community = self.belong[node]
                best_community = community
                best_gain = 0
                best_partition[community].nodes.remove(node)
                best_shared_links = 0
                for x, y, val in self.edges_of_nodes[node]:
                    if x == y:
                        continue
                    elif self.belong[y] == community:
                        best_shared_links += val
                best_partition[community].s_in -= 2 * \
                    (best_shared_links + self.w.get(node, 0))
                best_partition[community].s_tot -= self.ki[node]
                self.belong[node] = -1
                comm_mark = {}
                for neighbor in self.Neighbors(node):
                    ncomm = self.belong[neighbor]
                    if ncomm in comm_mark:
                        continue
                    comm_mark[ncomm] = True
                    shared_links = 0
                    for x, y, val in self.edges_of_nodes[node]:
                        if x == y:
                            continue
                        elif self.belong[y] == ncomm:
                            shared_links += val
                    gain = self.Modularity_Gain(
                        node, best_partition[ncomm], shared_links
                    )
                    if gain > best_gain:
                        best_community = ncomm
                        best_gain = gain
                        best_shared_links = shared_links

                best_partition[best_community].nodes.add(node)
                self.belong[node] = best_community
                best_partition[best_community].s_in += 2 * \
                    (best_shared_links + self.w.get(node, 0))
                best_partition[best_community].s_tot += self.ki.get(node, 0)
                if community != best_community:
                    Improved = True
            # print(idx, Improved)
            # idx += 1
            if not Improved:
                break
        return best_partition

    def second_phase(self, node, edges, partiton):
        nodes_ = set(partiton.keys())
        edges_ = {}
        for x, y, val in edges:
            ci = self.belong[x]
            cj = self.belong[y]
            edges_[(ci, cj)] = edges_.get((ci, cj), 0) + val
        nedges_ = [(x, y, z) for (x, y), z in edges_.items()]
        self.ki, self.w = {}, {}
        for node in nodes_:
            self.ki[node] = 0
            self.w[node] = 0

        self.edges_of_nodes = {}
        for x, y, val in nedges_:
            self.ki[x] += val
            self.ki[y] += val
            if x == y:
                self.w[x] += val
            if x not in self.edges_of_nodes:
                self.edges_of_nodes[x] = []
            if y not in self.edges_of_nodes:
                self.edges_of_nodes[y] = []

            self.edges_of_nodes[x].append((x, y, val))
            if x != y:
                self.edges_of_nodes[y].append((y, x, val))

        self.belong = {}
        return nodes_, nedges_

    def Solve(self):
        nnode, nedges = self.Nodes, [(x, y, 1) for x, y in self.edges]
        best_q, iternum = -1, 1
        while True:
            print('{} iterations '.format(iternum))
            iternum += 1
            partition = self.first_phase(nnode, nedges)
            Q = self.Modularity(partition)
            empty_keys = [k for k, v in partition.items() if len(v.nodes) == 0]
            for x in empty_keys:
                del partition[x]
            if self.actual_partiton:
                tempans = {}
                for k, v in partition.items():
                    tempans[k] = set()
                    for nod in v.nodes:
                        tempans[k].update(self.actual_partiton[nod])
                self.actual_partiton = tempans
            else:
                tempans = {}
                for k, v in partition.items():
                    tempans[k] = v.nodes
                self.actual_partiton = tempans
            print("Number of clusters: {}".format(len(partition)))
            if Q <= best_q:
                break
            nnode, nedges = self.second_phase(nnode, nedges, partition)
            best_q = Q

        return self.actual_partiton, best_q
