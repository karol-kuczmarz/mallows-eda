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
        restart_after_central_permutaition_fix,
    ):
        self.problem_size = problem_size - 1
        self.objective_function = lambda x: objective_function(x + 1)
        self.population_size = population_size
        self.selection_size = selection_size
        self.offspring_size = offspring_size
        self.n_iter = n_iter
        self.selection_function = selection_function
        self.restart_after_central_permutaition_fix = (
            restart_after_central_permutaition_fix
        )

    def evolve(self, disable_tqdm=False):
        population = Uniform(self.problem_size).sample_n(self.population_size)
        old_central_permutation = np.zeros(self.problem_size)
        central_permutation_repetitions = 0
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
                print(
                    f"Central permutation repetitions: {central_permutation_repetitions}"
                )

            if (central_permutation == old_central_permutation).all():
                central_permutation_repetitions += 1
                if (
                    central_permutation_repetitions
                    > self.restart_after_central_permutaition_fix
                ):
                    print("Applying shake procedure")
                    population_objectives = self.objective_function(population)
                    population = self.shake(
                        population[population_objectives.argmin(), :],
                        self.population_size,
                    )
                    central_permutation_repetitions = 0
            else:
                old_central_permutation = central_permutation
                central_permutation_repetitions = 0

        return central_permutation, dispersion_parameter

    def shake(self, permutation, population_size):
        population = np.tile(permutation, (population_size, 1))
        for i in range(population_size):
            for _ in range(5):
                idx = np.random.choice(self.problem_size)
                new_idx = np.random.choice(
                    np.arange(
                        np.max([0, idx - 5]), np.min([self.problem_size, idx + 5])
                    )
                )
                old_value = population[i, idx]
                population[i, idx : self.problem_size - 1] = population[
                    i, idx + 1 : self.problem_size
                ]
                population[i, new_idx + 1 : self.problem_size] = population[
                    i, new_idx : self.problem_size - 1
                ]
                population[i, new_idx] = old_value
        return population
