from src.utils.dataset_tools import get_training_testing_windows
from src.representations.lstm_autoencoder import LSTMAutoencoder
from src.evolution_algorithms.nsga2.nsga2 import NSGA2


def main():
    csv_file = 'data/dataset.csv'
    x_train, y_train, x_test, y_test = get_training_testing_windows(csv_file, sequence_length=200, train_size=0.8,
                                                                    size_train=100, size_test=100)

    train_test_data = {
        'epochs': 1,
        'batch_size': 32,
        'x_train': x_train,
        'y_train': y_train,
        'x_test': x_test,
        'y_test': y_test
    }

    # Same way we can use crossover parameters.
    mutation_parameters = {
        'mutation_number': 1,
        'unique': False,
        'lstm_random': True,
        'change_rate': 0.2
    }

    representation_object = LSTMAutoencoder(input_shape=(180, 6),
                                            output_shape=(20, 6),
                                            max_lstm_layers=5,
                                            train_test_data=train_test_data,
                                            mutation_parameters=mutation_parameters)

    ea = NSGA2(population_size=5, max_generations=50, num_objectives=3, optimization_directions=['min', 'min', 'min'], mutation_probability=0.5, representation_object=representation_object)

    ea.create_initial_population()

    ea.sort_population()

    ea.new_generation()


if __name__ == '__main__':
    main()




