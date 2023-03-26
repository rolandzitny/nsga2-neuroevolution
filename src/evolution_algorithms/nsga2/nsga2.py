"""
BASE INFO
"""
import random
from src.representations.representation import Representation


class NSGA2:
    def __init__(self, population_size, max_generations, num_objectives, optimization_directions, mutation_probability,
                 representation_object=Representation):
        """
        NSGA2

        :param population_size: Size of population.
        :param max_generations: Maximal number of generations.
        :param num_objectives: Number of NSGA2 optimization objectives.
        :param optimization_directions: Optimization directions for every optimization object,
         for example ['min', 'max', 'min'].
        :param representation_object: Representation object, with predefined abstract methods used in NSGA2.
        """
        self.representation_object = representation_object
        self.population_size = population_size
        self.max_generations = max_generations
        self.num_objectives = num_objectives
        self.mutation_probability = mutation_probability
        self.optimization_directions = optimization_directions

        self.population = None

    # noinspection PyArgumentList
    def create_initial_population(self):
        """
        Creates initial population using representation_object method.
        """
        self.population = self.representation_object.create_initial_population(self.population_size)

    def evaluate_population(self):
        """
        Evaluates every individual of population, with representation_object method.
        # TODO multithreading or multiprocessing

        :return: Evaluation results, for example [[obj1, obj2, obj3], [...], ...].
        """
        evaluation_results = []
        for i in range(self.population_size):
            evaluation_results.append(self.population[i].evaluate())

        return evaluation_results

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
        # Calculate the crowding distance for each solution in the front
        distances = [0] * len(front)
        for obj in range(self.num_objectives):
            sorted_front = sorted(front, key=lambda x: evaluation_results[x][obj])
            distances[0] = distances[-1] = float('inf')
            for i in range(1, len(front) - 1):
                distances[i] += (
                        (evaluation_results[sorted_front[i + 1]][obj] - evaluation_results[sorted_front[i - 1]][obj])
                        / (evaluation_results[sorted_front[-1]][obj] - evaluation_results[sorted_front[0]][obj])
                )

        # Return the sorted front based on crowding distance
        return [x for _, x in sorted(zip(distances, front), reverse=True)]

    def sort_population(self):
        """
        By combining non-dominated sorting and crowding distance sorting, NSGA-II is able to efficiently explore the
        Pareto front and find a diverse set of optimal solutions. This method only sorts individuals in population.
        """
        # Perform the fast non-dominated sort
        population_results = self.evaluate_population()
        fronts = self._fast_non_dominated_sort(population_results)

        # Sort each front based on crowding distance
        sorted_population = []
        for front in fronts:
            try:
                sorted_front = self._crowding_distance_sort(population_results, front)
                sorted_population.extend(sorted_front)

                # Stop sorting when the sorted population has reached the desired population size
                if len(sorted_population) >= self.population_size:
                    break
            except:
                pass

        # Truncate the sorted population to the desired size
        self.population = [self.population[i] for i in sorted_population[:self.population_size]]

    def new_generation(self):
        """
        Creating a new generation as follows:

        1. The first (best) individual is selected.
        2. This best individual performs its crossover method with each of the following individuals,
           from which two offspring are returned until the population size is filled.
        3. Subsequently, mutation is applied to the individuals with a certain probability.

        TODO Not every crossover method can return 2 offsprings -> future update
        """
        best_individual = self.population[0]
        new_generation = []
        others_number = (self.population_size // 2) + 1

        print("--------------------------------------------------------------------------------------------")
        for pop in self.population:
            pop.display_representation()

        for i in range(others_number):
            print(i + 1)
            offspring1, offspring2 = best_individual.crossover(self.population[i + 1])

            if random.random() < self.mutation_probability:
                offspring1 = offspring1.mutate()

            if random.random() < self.mutation_probability:
                offspring2 = offspring2.mutate()

            new_generation.append(offspring1)
            new_generation.append(offspring2)

        new_generation = new_generation[:self.population_size]
        self.population = new_generation

        print("")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        for pop in self.population:
            pop.display_representation()

    def run(self):
        """
        Initialize the population of LSTM Autoencoder representations

        Evaluate the population using the objective functions

        Sort the population into different Pareto fronts based on their non-dominance rank and crowding distance

        Select the solutions for the next generation using a combination of non-domination rank and crowding distance

        Perform genetic operations on the selected solutions to create the next generation

        Evaluate the new population and repeat steps 3-5 until the termination criteria are met
        """
        pass
