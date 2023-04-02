"""
NSGA2 (Non-dominated Sorting Genetic Algorithm 2) is a multi-objective optimization algorithm based on genetic
algorithms. NSGA-II uses non-dominated sorting to determine the fitness of individuals in the population,
rather than a single objective function. NSGA-II maintains a set of solutions that are not dominated by each other
in terms of multiple objectives, which is known as the Pareto front.

1. Initialize the population with random individuals.
2. Evaluate each individual based on multiple objectives.
3. Perform non-dominated sorting to identify the individuals that are not dominated by each other.
   This creates a set of Pareto fronts.
4. Assign a crowding distance to each individual within each front to promote diversity within the population.
5. Create new generation of population using evolutionary operators like selection, mutation and crossover.
6. Repeat 2-5 till termination criterion is met.

Class parameter example:

    nsga2_parameters = {
        'population_size': 4,
        'max_generations': 3,
        'num_objectives': 3,
        'optimization_directions': ['min', 'min', 'min'],
        'mutation_probability': 0.5,
        'use_multiprocessing': True
    }

"""
import math
import random
import multiprocessing
from src.representations.representation import Representation


def evaluate_individual(individual):
    """
    This function is used to map evaluation on every individual while using multiprocessing.

    :param individual: Individual of population.
    :return: Evaluation results.
    """
    return individual.evaluate()


class NSGA2:
    def __init__(self, nsga2_parameters, representation_object=Representation, logger=None):
        """
        NSGA2.
        Representation object needs to have implemented methods of Representation abstract class.

        :nsga2_parameters population_size: Size of population.
        :nsga2_parameters archive_size: Size of archive of elites with their evaluation results (solution, result)
        (the best solutions for the entire run of the algorithm).
        :nsga2_parameters max_generations: Maximal number of generations.
        :nsga2_parameters num_objectives: Number of NSGA2 optimization objectives.
        :nsga2_parameters optimization_directions: Optimization directions for every optimization object,
         for example ['min', 'max', 'min'].
        :nsga2_parameters mutation_probability: Probability of mutation of new population.
        :nsga2_parameters use_multiprocessing: True/False -> using multiprocessing on evaluation.
        :nsga2_parameters discard_individuals: True/False -> whether to remove individuals with 'nan'/'inf' evaluation
        values from population before sorting.
        :param representation_object: Representation object, with predefined abstract methods used in NSGA2.
        """
        self.representation_object = representation_object
        self.population_size = nsga2_parameters['population_size']
        self.archive_size = nsga2_parameters['archive_size']
        self.max_generations = nsga2_parameters['max_generations']
        self.num_objectives = nsga2_parameters['num_objectives']
        self.mutation_probability = nsga2_parameters['mutation_probability']
        self.optimization_directions = nsga2_parameters['optimization_directions']
        self.use_multiprocessing = nsga2_parameters['use_multiprocessing']
        self.discard_individuals = nsga2_parameters['discard_individuals']
        self.logger = logger

        self.population = []
        self.archive = []
        self.evaluation_results = []

    def _create_initial_population(self):
        """
        Creates initial population using representation_object method.
        """
        self.population = self.representation_object.create_initial_population(population_size=self.population_size)

    def _update_archive(self, pareto_optimal_solution_indexes):
        """
        Update archive of the best solutions yet obtained. Those solutions will be used in new generation.
        This procedure is used to preserve the best solutions so far in the population.

        :param pareto_optimal_solution_indexes: Pareto optimal front from sort_population method.
        """
        solutions = [self.population[i] for i in pareto_optimal_solution_indexes]
        eval_results = [self.evaluation_results[i] for i in pareto_optimal_solution_indexes]

        # TODO logging
        if self.logger is not None:
            for i in range(len(solutions)):
                self.logger.log_evaluation(individual_id=solutions[i].get_id(),
                                           mape=eval_results[i][0],
                                           num_params=eval_results[i][1],
                                           memory_usage=eval_results[i][2])

        archive_solutions = []
        archive_results = []
        for (sol, res) in self.archive:
            archive_solutions.append(sol)
            archive_results.append(res)

        all_solutions = solutions + archive_solutions
        all_results = eval_results + archive_results

        new_archive_pareto_front_indexes = self._fast_non_dominated_sort(all_results)[0]
        new_archive_solutions = [all_solutions[i] for i in new_archive_pareto_front_indexes]
        new_archive_results = [all_results[i] for i in new_archive_pareto_front_indexes]

        new_archive = []
        for i in range(len(new_archive_solutions)):
            new_archive.append((new_archive_solutions[i], new_archive_results[i]))

        # Fill new archive with solutions which are not included
        for i in range(len(all_solutions)):
            if all_solutions[i] not in new_archive_solutions:
                new_archive.append((all_solutions[i], all_results[i]))

        # Resize new archive to archive size
        new_archive = new_archive[:self.archive_size]
        self.archive = new_archive

    def _discard_individuals(self, evaluation_results):
        """
        If self.discard_individuals is True remove all individuals that have nan/inf values in evaluation result.

        :param evaluation_results: Evaluation results.
        :return: Cleared population, cleared evaluation results by indexes.
        """
        # Remove all individuals who have 'nan'/'inf' value in evaluation results
        remove_indexes = []
        for i in range(len(self.population)):
            if sum(evaluation_results[i]) == math.nan or sum(evaluation_results[i]) == math.inf:
                remove_indexes.append(i)

        cleared_population = [elem for i, elem in enumerate(self.population) if i not in remove_indexes]
        cleared_evaluation_results = [elem for i, elem in enumerate(evaluation_results) if i not in remove_indexes]

        return cleared_population, cleared_evaluation_results

    def _evaluate_population(self):
        """
        Evaluates every individual of population, with representation_object method evaluate.

        :return: Evaluation results, for example [[obj1, obj2, obj3], [...], ...].
        """
        evaluation_results = []

        # Multiprocessing
        if self.use_multiprocessing:
            num_processes = multiprocessing.cpu_count()
            ctx = multiprocessing.get_context('spawn')
            pool = ctx.Pool(processes=num_processes)
            results = pool.map(evaluate_individual, self.population)
            pool.close()

            # Sort results by population ID
            for individual in self.population:
                individual_name = individual.get_id()
                for result in results:
                    if individual_name == result[0]:
                        evaluation_results.append(result[1])
        # Sequential
        else:
            for i in range(self.population_size):
                evaluation_results.append(self.population[i].evaluate()[1])

        # Remove all individuals who have 'nan'/'inf' value in evaluation results
        if self.discard_individuals:
            cleared_population, cleared_evaluation_results = self._discard_individuals(evaluation_results)
            self.population = cleared_population
            evaluation_results = cleared_evaluation_results

        self.evaluation_results = evaluation_results

    def _fast_non_dominated_sort(self, evaluation_results):
        """
        The following code was generated with the help of ChatGPT, a language model trained by OpenAI.

        Non-dominated sorting procedure is used to classify the solutions in the population into different
        non-dominated fronts like this:

        1. For each solution in the population, fast_non_dominated_sort determines the set of solutions that dominate it
           (i.e., solutions that are better in all objectives) and the set of solutions that it dominates
           (i.e., solutions that are worse in at least one objective).

        2. The solutions that are not dominated by any other solution are assigned to the first non-dominated front.

        3. For each subsequent front, fast_non_dominated_sort repeats the same procedure, but only considers the
           solutions that have not yet been assigned to any previous front. The process continues until all solutions
           have been assigned to a non-dominated front.

        :param evaluation_results: Evaluation results from evaluate_population.
        :return: Fronts.
        """
        fronts = [[]]  # List of non-dominated fronts
        domination_counts = {}  # Number of solutions that dominate each solution
        dominated_solutions = {}  # Solutions dominated by each solution

        # For each solution, find the set of solutions that it dominates and the set of solutions that dominate it
        for i, result_i in enumerate(evaluation_results):
            domination_counts[i] = 0
            dominated_solutions[i] = set()
            for j, result_j in enumerate(evaluation_results):
                if i == j:
                    continue
                is_dominated = all(
                    (self.optimization_directions[k] == 'min' and result_i[k] <= result_j[k]) or
                    (self.optimization_directions[k] == 'max' and result_i[k] >= result_j[k])
                    for k in range(self.num_objectives)
                )
                if is_dominated:
                    dominated_solutions[i].add(j)
                elif all(
                        (self.optimization_directions[k] == 'min' and result_i[k] >= result_j[k]) or
                        (self.optimization_directions[k] == 'max' and result_i[k] <= result_j[k])
                        for k in range(self.num_objectives)
                ):
                    domination_counts[i] += 1

            # Add the solution to the first non-dominated front if it is not dominated by any other solution
            if domination_counts[i] == 0:
                fronts[0].append(i)

        # Assign solutions to subsequent non-dominated fronts
        i = 0
        while len(fronts[i]) > 0:
            next_front = []
            for sol in fronts[i]:
                for dominated_sol in dominated_solutions[sol]:
                    domination_counts[dominated_sol] -= 1
                    if domination_counts[dominated_sol] == 0:
                        next_front.append(dominated_sol)
            i += 1
            fronts.append(next_front)

        return fronts

    def _crowding_distance_sort(self, evaluation_results, front):
        """
        The following code was generated with the help of ChatGPT, a language model trained by OpenAI.

        Once the population has been sorted into non-dominated fronts, NSGA-II uses a crowding distance sorting
        procedure to select the best solutions for the next generation. The crowding_distance_sort function is used
        to perform this sorting. Here is how it works:

        1. For each non-dominated front, crowding_distance_sort calculates the crowding distance for each solution.
           The crowding distance measures the density of solutions around a particular solution in the objective space.

        2. The solutions in each front are then sorted based on their crowding distance, with solutions that have a
           larger crowding distance being preferred.

        3. The top solutions from each front are then selected to form the next generation.

        :param evaluation_results: Evaluation results from evaluate_population.
        :param front: One front from list of fronts returned from _fast_non_dominated_sort.
        :return: Sorted front.
        """
        # Initialize the distances list
        distances = [0] * len(front)

        # Calculate the crowding distance for each objective
        for obj in range(self.num_objectives):
            # Sort the front based on the current objective
            sorted_front = sorted(front, key=lambda x: evaluation_results[x][obj])

            # Update the crowding distances for each solution
            for i in range(1, len(front) - 1):
                if evaluation_results[sorted_front[-1]][obj] == evaluation_results[sorted_front[0]][obj]:
                    distances[i] += 0
                else:
                    distances[i] += (
                        (evaluation_results[sorted_front[i + 1]][obj] - evaluation_results[sorted_front[i - 1]][obj])
                        / (evaluation_results[sorted_front[-1]][obj] - evaluation_results[sorted_front[0]][obj])
                    )

        # Sort the front based on the crowding distances
        return [x for _, x in sorted(zip(distances, front), reverse=True)]

    def _sort_population(self):
        """
        By combining non-dominated sorting and crowding distance sorting, NSGA-II is able to efficiently explore the
        Pareto front and find a diverse set of optimal solutions. This method only sorts individuals in population.
        """
        # Perform the fast non-dominated sort
        fronts = self._fast_non_dominated_sort(self.evaluation_results)

        # Update archive of pareto optimal solutions
        pareto_optimal_solution_indexes = fronts[0]
        self._update_archive(pareto_optimal_solution_indexes)

        # Sort each front based on crowding distance
        sorted_population = []
        for i in range(len(fronts)):
            sorted_front = self._crowding_distance_sort(self.evaluation_results, fronts[i])

            # Save best individuals and keep them mutated in new generation
            if i == 1:
                self.first_front_individuals = [self.population[i] for i in sorted_front]

            sorted_population.extend(sorted_front)

            # Stop sorting when the sorted population has reached the desired population size
            if len(sorted_population) >= self.population_size:
                break

        # Truncate the sorted population to the desired size
        self.population = [self.population[i] for i in sorted_population[:self.population_size]]
        self.evaluation_results = [self.evaluation_results[i] for i in sorted_population[:self.population_size]]

    def _new_generation(self):
        """
        Creating a new generation as follows:

        1. First individuals in new population are individuals which are from archive.
        2. Rank-based selection: archive is joined with current sorted population,
           so first individual has highest chance and last individual has lowest chance to be chosen.
        3. Choose randomly (with probabilities) 2 individuals and do crossover. Result will be append to new generation.
        4. At the end mutate every individual in new generation, this step needs to be last because we dont want to
           change individuals from non dominated front before crossover.

        TODO Not every crossover method can return just 1 offsprings -> future update
        """
        new_generation = []

        # Get archive individuals
        archive_individuals = []
        for (individual, _) in self.archive:
            archive_individuals.append(individual)

        all_possible_individuals = []
        all_possible_individuals.extend(archive_individuals)

        # Create list of all unique individuals from archive and current population.
        for individual in self.population:
            if individual not in all_possible_individuals:
                all_possible_individuals.append(individual)

        total_rank = sum(range(len(all_possible_individuals) + 1))
        ranks = [i + 1 for i in range(len(all_possible_individuals))]
        probabilities = [(len(all_possible_individuals) - r + 1) / total_rank for r in ranks]

        # Make sure that first elements of new generation are individuals from archive.
        new_generation.extend(archive_individuals)

        # While population is not full
        while len(new_generation) < self.population_size:
            parent1, parent2 = random.choices(all_possible_individuals, weights=probabilities, k=2)
            # check if the two selected individuals are the same and sample again if they are
            while parent1 == parent2:
                parent1, parent2 = random.choices(all_possible_individuals, weights=probabilities, k=2)

            # Append new offspring from crossover
            new_generation.append(parent1.crossover(parent2))

        # Mutate every individual in new generation
        for individual in new_generation:
            if random.random() < self.mutation_probability:
                individual.mutate()

        self.population = new_generation

    def run(self):
        self._create_initial_population()

        for i in range(self.max_generations):

            if self.logger is not None:
                self.logger.log_generation(i)

            self._evaluate_population()
            self._sort_population()
            self._new_generation()

            if self.logger is not None:
                self.logger.log_mutations()
                self.logger.reset_mutation_logs()
