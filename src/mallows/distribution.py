import numpy as np


class Mallows:
    def __init__(self, sigma_0, theta, metric):
        self.sigma_0 = sigma_0
        self.theta = theta
        self.metric = metric
        self.n = len(sigma_0)
        self.normalization_constant = self._compute_normalization_constant()
        self.V = self._compute_V()

    def sample(self):
        possible_values = list(range(self.n))
        pi = np.zeros(self.n, dtype=np.int64)
        for j in range(0, self.n - 1):
            r_j_dist = np.exp(-self.theta * np.arange(0, self.n - j)) / self.V[j]
            r_j = np.random.choice(np.arange(0, self.n - j), p=r_j_dist)
            pi[j] = possible_values[r_j]
            del possible_values[r_j]
        pi[self.n - 1] = possible_values[0]
        sigma = pi[np.argsort(self.sigma_0)]
        return sigma

    def sample_n(self, m):
        V_values = np.zeros((m, self.n), dtype=np.int64)
        for j in range(0, self.n - 1):
            r_j_dist = np.exp(-self.theta * np.arange(0, self.n - j)) / self.V[j]
            V_values[:, j] = np.random.choice(
                np.arange(0, self.n - j), size=m, p=r_j_dist, replace=True
            )
        pi = np.zeros((m, self.n), dtype=np.int64)
        identities = np.tile(np.arange(0, self.n), (m, 1))
        identities_not_modified = np.tile(np.arange(0, self.n), (m, 1))
        for j in range(0, self.n - 1):
            pi[:, j] = identities[np.arange(0, m), V_values[:, j]]
            identities[
                (identities_not_modified.T >= V_values[:, j]).T
                & (identities_not_modified.T < self.n - j - 1).T
            ] = identities[
                (identities_not_modified.T >= V_values[:, j] + 1).T
                & (identities_not_modified.T < self.n - j).T
            ]
        pi[:, self.n - 1] = identities[:, 0]
        return pi[:, np.argsort(self.sigma_0)]

    def probability(self, sigma):
        return (
            np.exp(-self.theta * self.metric(self.sigma_0, sigma))
            / self.normalization_constant
        )

    def _compute_normalization_constant(self):
        prod = 1.0
        for j in range(1, self.n):
            prod *= (1 - np.exp(-self.theta * (self.n - j + 1))) / (
                1 - np.exp(-self.theta)
            )
        return prod

    def _compute_V(self):
        return (1 - np.exp(-self.theta * (self.n - np.arange(1, self.n) + 1))) / (
            1 - np.exp(-self.theta)
        )


class Uniform:
    def __init__(self, n):
        self.n = n
        self.rng = np.random.default_rng()

    def sample(self):
        return self.rng.permutation(self.n)

    def sample_n(self, m):
        return np.array([self.sample() for _ in range(m)])
