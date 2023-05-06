import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler


OPTIMIZATION_DIRECTIONS = ['min', 'min', 'min']


def plot_pareto_front(pareto_front):
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')

    # Extract values for mse, trainable parameters, and memory usage from pareto front
    mse_pf, ntrain_pf, mem_pf = pareto_front[:, 0], pareto_front[:, 1], pareto_front[:, 2]

    # Scale the data using MinMaxScaler
    scaler = MinMaxScaler()
    X = np.column_stack((mse_pf, ntrain_pf, mem_pf))
    X = scaler.fit_transform(X)
    mse_pf, ntrain_pf, mem_pf = X[:, 0], X[:, 1], X[:, 2]

    # Plot scatter points
    ax.scatter(mse_pf, ntrain_pf, mem_pf, c='b', alpha=0.5, s=40)

    # Set axis labels
    ax.set_xlabel('MSE')
    ax.set_ylabel('Trainable parameters (k)')
    ax.set_zlabel('Memory usage (k)')

    # Set tick labels for trainable parameters
    ntrain_labels = np.linspace(min(pareto_front[:, 1]), max(pareto_front[:, 1]), 5)/1000
    ntrain_labels_scaled = scaler.transform(np.column_stack((np.zeros_like(ntrain_labels), ntrain_labels*1000, np.zeros_like(ntrain_labels))))[:, 1]
    ax.set_yticks(ntrain_labels_scaled)
    ax.set_yticklabels(['{:,.0f}'.format(label) for label in ntrain_labels])

    # Set tick labels for MSE
    mse_labels = np.linspace(min(pareto_front[:, 0]), max(pareto_front[:, 0]), 5)
    mse_labels_scaled = scaler.transform(np.column_stack((mse_labels, np.zeros_like(mse_labels), np.zeros_like(mse_labels))))[:, 0]
    ax.set_xticks(mse_labels_scaled)
    ax.set_xticklabels(['{:,.2f}'.format(label) for label in mse_labels])

    # Set tick labels for memory usage
    mem_labels = np.linspace(min(pareto_front[:, 2]), max(pareto_front[:, 2]), 5)/1000
    mem_labels_scaled = scaler.transform(np.column_stack((np.zeros_like(mem_labels), np.zeros_like(mem_labels), mem_labels*1000)))[:, 2]
    ax.set_zticks(mem_labels_scaled)
    ax.set_zticklabels(['{:,.2f}'.format(label) for label in mem_labels])

    # Show plot
    plt.show()


def non_dom_sort(data):
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

    evaluation_results = data
    num_objectives = 3
    optimization_directions = OPTIMIZATION_DIRECTIONS

    # For each solution, find the set of solutions that it dominates and the set of solutions that dominate it
    for i, result_i in enumerate(evaluation_results):
        domination_counts[i] = 0
        dominated_solutions[i] = set()
        for j, result_j in enumerate(evaluation_results):
            if i == j:
                continue
            is_dominated = all(
                (optimization_directions[k] == 'min' and result_i[k] <= result_j[k]) or
                (optimization_directions[k] == 'max' and result_i[k] >= result_j[k])
                for k in range(num_objectives)
            )
            if is_dominated:
                dominated_solutions[i].add(j)
            elif all(
                    (optimization_directions[k] == 'min' and result_i[k] >= result_j[k]) or
                    (optimization_directions[k] == 'max' and result_i[k] <= result_j[k])
                    for k in range(num_objectives)
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


def parse_log_line(line):
    # split the line by commas
    parts = line.strip().split(',')

    # extract the required values
    mape = float(parts[1].split('=')[1])
    model_params = int(parts[2].split('=')[1])
    memory_usage = float(parts[3].split('=')[1])

    # return the values as a tuple
    return mape, model_params, memory_usage


def load_log_file(filepath):
    generations = []
    with open(filepath, 'r') as f:
        lines = f.readlines()
        start_idx = 0
        for i, line in enumerate(lines):
            if line.startswith("GENERATION"):
                generations.append(lines[start_idx:i])
                start_idx = i
        generations.append(lines[start_idx:])

    generations = generations[1:]

    pareto_fronts_lines = []
    pareto_fronts_values = []
    mutations = []

    for gen in generations:
        gen.pop(0)
        mutations.append(gen.pop())
        pareto_fronts_lines.append(gen)

    for front in pareto_fronts_lines:
        for row in front:
            mape, params, usage = parse_log_line(row)
            pareto_fronts_values.append([mape, params, usage])

    return pareto_fronts_lines, mutations, pareto_fronts_values


def create_pareto_fronts_folder(folder_path):
    pareto_fronts = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.txt'):
            _, _, res = load_log_file(os.path.join(folder_path, file_name))
            individuals = np.array(res)
            pareto_front = non_dom_sort(individuals)
            pareto_individuals = [res[i] for i in pareto_front[0]]
            pareto_fronts.append(pareto_individuals)

    pareto_front_combined = np.concatenate(pareto_fronts)
    pareto_front_final = non_dom_sort(pareto_front_combined)
    pareto_individuals = [pareto_front_combined[i] for i in pareto_front_final[0]]
    pareto_individuals = np.array(pareto_individuals)
    plot_pareto_front(pareto_individuals)
