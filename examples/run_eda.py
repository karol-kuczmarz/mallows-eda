from mallows import config as mallows_config
from mallows.distribution import Mallows
from mallows.eda import EDA
from mallows.selection import (adaptation_roulette_selection,
                               exponential_ranking_selection,
                               get_linear_ranking_selection, top_k_selection)
from mallows.tsp_utils import get_tsp_problem, plot_solution
import wandb
import numpy as np
from tqdm import trange, tqdm
import json

name_to_selection_function = {
    "adaptation_roulette_selection": adaptation_roulette_selection,
    "exponential_ranking_selection": exponential_ranking_selection,
    "get_linear_ranking_selection": get_linear_ranking_selection,
    "top_k_selection": top_k_selection
}

def run_eda(config, verbose=False):

    coords, dist_matrix, objective_function, optimal_solution = get_tsp_problem(
        mallows_config.DATA_DIR / "tsp", config["problem_name"]
    )
    wandb_run = wandb.init(project="mallows-eda",
               config={ "problem_name": config["problem_name"],
                        "problem_size": config["problem_size"],
                        "population_size": config["population_size"],
                        "selection_size": config["selection_size"],
                        "offspring_size": config["offspring_size"],
                        "n_iter": config["n_iter"],
                        "selection_function": config["selection_function"].__name__,
                        "restart_after_central_permutaition_fix": config["restart_after_central_permutaition_fix"],
                        "seed": config["seed"],
                        "optimal_solution": optimal_solution if optimal_solution is not None else 'N/A',
                        "optimal_solution_objective": objective_function(optimal_solution[1:].reshape(1, -1)) if optimal_solution is not None else 'N/A',
                        "series": config["series"] if "series" in config else "N/A"
                        })

    eda = EDA(
        coords.shape[0],
        objective_function,
        config["population_size"],
        config["selection_size"],
        config["offspring_size"],
        config["n_iter"],
        config["selection_function"],
        config["restart_after_central_permutaition_fix"],
        wandb_run
    )

    center_permutation, dispersion_parameter, best = eda.evolve(disable_tqdm=True)

    if verbose:
        plot_solution(best + 1, "Best found solution", coords, dist_matrix)

    wandb_run.finish()


def run_experiment_series(config, n_runs):
    np.random.seed(config["seed"])
    for _ in range(n_runs):
        run_eda(config)

# bays29_config = {
#     "problem_name": "bays29",
#     "problem_size": 29,
#     "population_size": 29000,
#     "selection_size": 2900,
#     "offspring_size": 28999,
#     "n_iter": 5000,
#     "selection_function": top_k_selection,
#     "restart_after_central_permutaition_fix": 250,
#     "seed": 42
# }

# bays29_config = {
#     "problem_name": "bays29",
#     "problem_size": 29,
#     "population_size": 2900,
#     "selection_size": 290,
#     "offspring_size": 2899,
#     "n_iter": 10000,
#     "selection_function": top_k_selection,
#     "restart_after_central_permutaition_fix": 100,
#     "seed": 42
# }
        
# bays29_config = {
#     "problem_name": "bays29",
#     "problem_size": 29,
#     "population_size": 290,
#     "selection_size": 29,
#     "offspring_size": 289,
#     "n_iter": 100000,
#     "selection_function": top_k_selection,
#     "restart_after_central_permutaition_fix": 100,
#     "seed": 42
# }
        
# bays29_config = {
#     "problem_name": "bays29",
#     "problem_size": 29,
#     "population_size": 290,
#     "selection_size": 29,
#     "offspring_size": 289,
#     "n_iter": 100000,
#     "selection_function": top_k_selection,
#     "restart_after_central_permutaition_fix": 250,
#     "seed": 42
# }
        
# bays29_config = {
#     "problem_name": "bays29",
#     "problem_size": 29,
#     "population_size": 29000,
#     "selection_size": 2900,
#     "offspring_size": 28999,
#     "n_iter": 10000,
#     "selection_function": top_k_selection,
#     "restart_after_central_permutaition_fix": 100,
#     "seed": 42,
#     "series": "E"
# }

experiments_per_config = 5
configs = json.load(open("./experiments/eda_configs.json", "r"))
for config in tqdm(configs):
    config["selection_function"] = name_to_selection_function[config["selection_function"]]
    run_experiment_series(config, experiments_per_config)