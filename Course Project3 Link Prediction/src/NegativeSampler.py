import numpy as np
from random import shuffle
import time
from tqdm import tqdm


class NegativeSampler:
    def __init__(self, distributions, verbose=False):
        starter = time.time()
        self.Keys = list(distributions.keys())
        totsum = sum(distributions.values())
        self.SampleLen = int(2e7)
        self.verbose = verbose

        if self.verbose:
            print('[INFO] Initializing Sampler...')

        if self.verbose:
            print('[INFO] Calculating the Distributions...')

        Curr = 0
        self.Bk = []
        for k in self.Keys:
            Curr = Curr + self.SampleLen * distributions[k] / totsum
            self.Bk.append(round(Curr))
        self.Bk[-1] = self.SampleLen

        if self.verbose:
            print('[INFO] Filling the Samples...')

        self.Samples = []
        idx = 0
        Iters = tqdm(
            range(self.SampleLen)
        ) if self.verbose else range(self.SampleLen)
        for i in Iters:
            if i == self.Bk[idx]:
                idx += 1
            self.Samples.append(self.Keys[idx])

        shuffle(self.Samples)
        self.currpos = 0

        if self.verbose:
            print('[INFO] Sampler Init Done')
            print('[INFO] {} seconds Used'.format(time.time() - starter))

    def Sample(self, nums):
        currres = self.SampleLen - self.currpos
        ans = []
        while nums >= currres:
            ans.append(self.Samples[self.currpos:])
            self.currpos = 0
            nums -= currres
            currres = self.SampleLen
            shuffle(self.Samples)

        if nums > 0:
            ans.append(self.Samples[self.currpos: self.currpos + nums])
            self.currpos += nums
        return np.concatenate(ans)


if __name__ == '__main__':
    A = NegativeSampler({
        1: 12234,
        2: 2364,
        5: 2131,
    })
    starter = time.time()
    print(A.Sample(20))
    print(A.Sample(40))
    print(A.Sample(10000))
    print(time.time() - starter)
