def top_k_selection(population_objectives, k):
    return population_objectives.argsort()[:k]
