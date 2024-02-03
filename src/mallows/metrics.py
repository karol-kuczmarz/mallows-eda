import numpy as np

class KendallTau:
    def __init__(self, n):
        self.n = n

    def __call__(self, p, q):
        q_inv = np.argsort(q)
        pq_inv = p[q_inv]
        value = 0
        for i in range(self.n):
            value += np.sum(pq_inv[i+1:] < pq_inv[i])
        return value
