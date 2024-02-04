from tqdm import tqdm

from mallows.distribution import Uniform


class EDA:
    def __init__(
        self,
        problem_size,
        objective_function,
        population_size,
        selection_size,
        offspring_size,
        n_iter,
    ):
        self.problem_size = problem_size
        self.objective_function = objective_function
        self.population_size = population_size
        self.selection_size = selection_size
        self.offspring_size = offspring_size
        self.n_iter = n_iter

    def evolve(self, disable_tqdm=False):
        population = Uniform(self.problem_size).sample_n(self.population_size)
        population_objectives = self.objective_function(population)

        for i in tqdm(range(self.n_iter), disable=disable_tqdm):
            pass
