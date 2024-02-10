import numpy as np
from tqdm import tqdm

from mallows.distribution import Mallows, Uniform
from mallows.metrics import KendallTau
from mallows.utils import estimate_mean, estimate_theta


class EDA:
    def __init__(
        self,
        problem_size,
        objective_function,
        population_size,
        selection_size,
        offspring_size,
        n_iter,
        selection_function,
    ):
        self.problem_size = problem_size - 1
        self.objective_function = lambda x: objective_function(x + 1)
        self.population_size = population_size
        self.selection_size = selection_size
        self.offspring_size = offspring_size
        self.n_iter = n_iter
        self.selection_function = selection_function

    def evolve(self, disable_tqdm=False):
        population = Uniform(self.problem_size).sample_n(self.population_size)
        for i in tqdm(range(self.n_iter), disable=disable_tqdm):
            population_objectives = self.objective_function(population)
            selected_indices = self.selection_function(
                population_objectives, self.selection_size
            )
            selected_population = population[selected_indices]
            central_permutation = estimate_mean(selected_population)
            dispersion_parameter = estimate_theta(
                selected_population, central_permutation
            )
            # dispersion_parameter = 0.01
            offspring = Mallows(
                central_permutation, dispersion_parameter, KendallTau(self.problem_size)
            ).sample_n(self.offspring_size)
            population = np.concatenate(
                [
                    population[np.argmin(population_objectives), :].reshape(1, -1),
                    offspring,
                ],
                axis=0,
            )
            if i % 100 == 0:
                print(
                    f"Generation {i} - Best: {population_objectives.min()}, Theta: {dispersion_parameter}"
                )
                print(f"Generation {i} - Average: {population_objectives.mean()}")
                print(
                    f"Generation {i} - Parents average: {population_objectives[selected_indices].mean()}"
                )
                print(
                    f"Generation {i} - Best individual repeats: {(population_objectives.argmin() == selected_indices).sum()}"
                )
                print(
                    f"Central permutation objective: {self.objective_function(central_permutation.reshape(1,-1))[0]}"
                )

        return central_permutation, dispersion_parameter
