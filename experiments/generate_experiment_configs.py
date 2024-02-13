from mallows.selection import (adaptation_roulette_selection,
                               exponential_ranking_selection,
                               get_linear_ranking_selection, top_k_selection)
import json

def generate_config(problem_name, problem_size, population_size, selection_size, offspring_size,
                    n_iter, restart_after_central_permutaition_fix, seed, series, selection_function="top_k_selection"):
    return {
        "problem_name": problem_name,
        "problem_size": problem_size,
        "population_size": population_size,
        "selection_size":  selection_size,
        "offspring_size": offspring_size,
        "n_iter": n_iter,
        "selection_function": selection_function,
        "restart_after_central_permutaition_fix": restart_after_central_permutaition_fix,
        "seed": 42,
        "series": series
    }

def generate_experiment(problem_name, problem_size):
    configs = []
    letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
    population_sizes = [problem_size * 10**(i+1) for i in range(3)]
    selection_sizes = [population_size // 10 for population_size in population_sizes]
    offspring_sizes = [population_size - 1 for population_size in population_sizes]
    n_iters = 10**5
    restart_after_central_permutaition_fix = [100, 250]
    for i in range(3):
        for j, restart_value in enumerate(restart_after_central_permutaition_fix):
            configs.append(generate_config(problem_name, problem_size, population_sizes[i], selection_sizes[i],
                                        offspring_sizes[i], n_iters, restart_value, 42, letters[2*i+j]))
    return configs

problems = [("bayg29", 29), ("bays29", 29), ("kroA100", 100), ("kroA150", 150), ("kroA200", 200)]

for problem_name, problem_dim in problems:
    configs = generate_experiment(problem_name, problem_dim)

    json.dump(configs, open(f"./experiments/{problem_name}_configs.json", "w"))