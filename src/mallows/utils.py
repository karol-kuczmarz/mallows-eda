import numpy as np


def estimate_mean(samples):
    samples = samples.copy() + 1
    pi = np.mean(samples, axis=0)
    permutation = np.argsort(pi)
    return permutation


def estimate_theta(samples, sigma_0):
    def f(theta, V, n):
        val = 0.0
        val += (n - 1.0) / (np.exp(theta) - 1)
        val -= np.sum(
            (n - np.arange(1, n) + 1) / (np.exp(theta * (n - np.arange(1, n) + 1)) - 1)
        )
        val -= V
        return val

    def f_prime(theta, n):
        val = 0.0
        val -= (n - 1) * (np.exp(theta)) / (np.exp(theta) - 1) ** 2
        val += np.sum(
            (n - np.arange(1, n) + 1) ** 2.0
            * (np.exp(theta * (n - np.arange(1, n) + 1)))
            / (np.exp(theta * (n - np.arange(1, n) + 1)) - 1) ** 2.0
        )
        return val

    def newton_raphson(f, f_prime, x_0=0.01, tol=1e-5, max_iter=1000):
        x = x_0
        for i in range(max_iter):
            f_x = f(x)
            x_new = x - f_x / f_prime(x)
            if np.abs(f_x) < tol:
                return x_new
            x = x_new
        return x

    sigma_0_inv = np.argsort(sigma_0)
    samples_sigma_0_inv = samples[:, sigma_0_inv]
    V_hat = np.zeros(sigma_0.shape[-1] - 1)
    for i in range(sigma_0.shape[-1] - 1):
        V_hat[i] = (
            np.sum(samples_sigma_0_inv[:, i + 1 :].T < samples_sigma_0_inv[:, i])
            / samples.shape[-2]
        )
    V = np.sum(V_hat)
    theta = newton_raphson(
        lambda x: f(x, V, sigma_0.shape[-1]),
        lambda x: f_prime(x, sigma_0.shape[-1]),
    )
    return theta
