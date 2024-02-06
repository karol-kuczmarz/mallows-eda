from mallows import config
from mallows.distribution import Mallows
from mallows.eda import EDA
from mallows.selection import top_k_selection
from mallows.tsp_utils import get_tsp_problem, plot_solution

problem_size = 29
population_size = (problem_size - 1) * 100
selection_size = (problem_size - 1) * 50
offspring_size = (problem_size - 1) * 100 - 1
n_iter = 100 * (problem_size - 1)

coords, dist_matrix, objective_function, optimal_solution = get_tsp_problem(
    config.DATA_DIR / "tsp", "bayg29"
)
eda = EDA(
    coords.shape[0],
    objective_function,
    population_size,
    selection_size,
    offspring_size,
    n_iter,
    top_k_selection,
)
center_permutation, dispersion_parameter = eda.evolve()
