import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from src.utils.logger import Logger
from src.utils.dataset_tools import get_training_testing_windows
from src.representations.lstm_autoencoder import LSTMAutoencoder
from src.evolution_algorithms.nsga2.nsga2 import NSGA2


def main():
    csv_file = 'data/dataset.csv'
    x_train, y_train, x_test, y_test = get_training_testing_windows(csv_file, sequence_length=200, train_size=0.8,
                                                                    size_train=100, size_test=100)

    x_test = x_train[:10]
    y_test = y_train[:10]

    train_test_data = {
        'epochs': 10,
        'batch_size': 32,
        'x_train': x_train,
        'y_train': y_train,
        'x_test': x_test,
        'y_test': y_test
    }

    # Same way we can use crossover parameters.
    mutation_parameters = {
        'mutation_number': 1,
        'unique': True,
        'lstm_random': True,
        'max_lstm_layers': 5,
        'change_rate': 0.2
    }

    mutation_function = ['LSTM1_ADD',
                         'LSTM1_REMOVE',
                         'LSTM2_ADD',
                         'LSTM2_REMOVE',
                         'DENSE_ACT',
                         'LSTM_UNITS',
                         'LSTM_ACT',
                         'LSTM_REC_ACT',
                         'LSTM_DROPOUT',
                         'LSTM_BATCHNORM']

    nsga2_parameters = {
        'population_size': 6,
        'max_generations': 3,
        'num_objectives': 3,
        'optimization_directions': ['min', 'min', 'min'],
        'mutation_probability': 0.5,
        'use_multiprocessing': True
    }

    representation_object = LSTMAutoencoder(input_shape=(180, 6),
                                            output_shape=(20, 6),
                                            train_test_data=train_test_data,
                                            mutation_methods=mutation_function,
                                            mutation_parameters=mutation_parameters)

    ea = NSGA2(nsga2_parameters=nsga2_parameters, representation_object=representation_object)

    ea.run()


if __name__ == '__main__':
    logger = Logger()
    logger.log("Starting main program...")
    # Nromal = 2
    # GPU = 12
    # TPU = 40
    main()
    logger.log('END')




