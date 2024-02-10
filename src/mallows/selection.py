import numpy as np


def top_k_selection(population_objectives, k):
    return population_objectives.argsort()[:k]


def get_linear_ranking_selection(alpha, beta):
    assert 0 <= alpha and alpha <= beta and beta <= 2 and alpha + beta <= 2

    def linear_ranking_selection(population_objectives, k):
        N = population_objectives.shape[0]
        ranks = np.argsort(population_objectives)[::-1]
        probabilities = (alpha + (ranks / (N - 1)) * (beta - alpha)) / N
        selected_indices = np.random.choice(N, k, replace=True, p=probabilities)
        return selected_indices

    return linear_ranking_selection


def exponential_ranking_selection(population_objectives, k):
    N = population_objectives.shape[0]
    ranks = np.argsort(population_objectives)[::-1]
    probabilities = 1 - np.exp(-ranks)
    probabilities /= probabilities.sum()
    selected_indices = np.random.choice(N, k, replace=True, p=probabilities)
    return selected_indices


def adaptation_roulette_selection(population_objectives, k):
    N = population_objectives.shape[0]
    worst = population_objectives.max()
    probabilities = worst - population_objectives
    probabilities /= probabilities.sum()
    selected_indices = np.random.choice(N, k, replace=True, p=probabilities)
    return selected_indices
