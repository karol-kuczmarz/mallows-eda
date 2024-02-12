from mallows import config
from mallows.distribution import Mallows
from mallows.eda import EDA
from mallows.selection import (adaptation_roulette_selection,
                               exponential_ranking_selection,
                               get_linear_ranking_selection, top_k_selection)
from mallows.tsp_utils import get_tsp_problem, plot_solution
import wandb

# problem_size = 14
# population_size = problem_size * 1000
# # population_size = problem_size * 10
# selection_size = problem_size * 100
# # selection_size = problem_size
# offspring_size = population_size - 1
# n_iter = problem_size * 1000
# selection_function = top_k_selection
# restart_after_central_permutaition_fix = 200

def run_eda(problem_name, problem_size, population_size, selection_size, offspring_size, n_iter,
            selection_function, restart_after_central_permutaition_fix, verbose=False):

    coords, dist_matrix, objective_function, optimal_solution = get_tsp_problem(
        config.DATA_DIR / "tsp", problem_name
    )
    if optimal_solution is not None:
        print(objective_function(optimal_solution[1:].reshape(1, -1)))

    wandb_run = wandb.init(project="mallows-eda",
               config={ "problem_name": problem_name,
                        "problem_size": problem_size,
                        "population_size": population_size,
                        "selection_size": selection_size,
                        "offspring_size": offspring_size,
                        "n_iter": n_iter,
                        "selection_function": selection_function.__name__,
                        "restart_after_central_permutaition_fix": restart_after_central_permutaition_fix
                        })


    eda = EDA(
        coords.shape[0],
        objective_function,
        population_size,
        selection_size,
        offspring_size,
        n_iter,
        selection_function,
        restart_after_central_permutaition_fix,
        wandb_run
    )
    center_permutation, dispersion_parameter, best = eda.evolve()

    if verbose:
        plot_solution(best + 1, "Best found solution", coords, dist_matrix)


run_eda("burma14", 14, 14000, 1400, 13999, 140, top_k_selection, 200, True)