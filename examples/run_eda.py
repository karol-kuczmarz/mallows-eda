from mallows import config
from mallows.distribution import Mallows
from mallows.eda import EDA
from mallows.selection import (adaptation_roulette_selection,
                               exponential_ranking_selection,
                               get_linear_ranking_selection, top_k_selection)
from mallows.tsp_utils import get_tsp_problem

problem_size = 14
population_size = problem_size * 1000
selection_size = problem_size * 100
offspring_size = population_size - 1
n_iter = problem_size * 1000
selection_function = top_k_selection

coords, dist_matrix, objective_function, optimal_solution = get_tsp_problem(
    config.DATA_DIR / "tsp", "burma14"
)
eda = EDA(
    coords.shape[0],
    objective_function,
    population_size,
    selection_size,
    offspring_size,
    n_iter,
    selection_function,
)
center_permutation, dispersion_parameter = eda.evolve()
