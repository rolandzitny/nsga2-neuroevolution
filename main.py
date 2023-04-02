import os
from src.utils.logger import LSTMAutoencoderLogger
from src.utils.dataset_tools import get_training_testing_windows
from src.representations.lstm_autoencoder import LSTMAutoencoder
from src.evolution_algorithms.nsga2.nsga2 import NSGA2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# NSGA2 parameters
MAX_GENERATIONS = 3
POPULATION_SIZE = 10
ARCHIVE_SIZE = 5                # Size of archive of the best solutions yet obtained.
NUM_OBJECTIVES = 3
OPTIMIZATION_DIRECTIONS = ['min', 'min', 'min']
MUTATION_PROBABILITY = 0.9
USE_MULTIPROCESSING = False      # Multiprocessing for evaluation.
DISCARD_INDIVIDUALS = True      # Discard individuals with nan/info evaluation results

# LSTM Autoencoder parameters
MUTATION_NUMBER = 1             # Number of executed mutations.
UNIQUE = True                   # If number of mutation > 1, choose unique methods, max 10.
LSTM_RANDOM = True              # Initialize new LSTM layer at random.
MAX_LSTM_LAYERS = 2
CHANGE_RATE = 0.3               # Rate of units and dropout change, 0.2 -> range(-20% , +20%)

LSTM_UNITS = (8, 16)
LSTM_ACTIVATION = ['sigmoid', 'tanh', 'relu', 'softmax', 'linear']
LSTM_REC_ACTIVATION = ['sigmoid', 'hard_sigmoid', 'tanh']
DENSE_ACTIVATION = ['sigmoid', 'tanh', 'softmax', 'linear']
DROPOUT = (0.0, 0.5)
BATCH_NORM = [True, False]

TRAIN_EPOCHS = 1
BATCH_SIZE = 32

LSTM1_ADD_PROB = 0.1
LSTM1_REMOVE_PROB = 0.1
LSTM2_ADD_PROB = 0.1
LSTM2_REMOVE_PROB = 0.1
DENSE_ACT_PROB = 0.8
LSTM_UNITS_PROB = 0.1
LSTM_ACT_PROB = 0.1
LSTM_REC_ACT_PROB = 0.1
LSTM_DROPOUT_PROB = 0.1
LSTM_BATCHNORM_PROB = 0.1


def log_all_parameters(logger):
    logger.log(f'MAX_GENERATIONS             : {MAX_GENERATIONS}')
    logger.log(f'POPULATION_SIZE             : {POPULATION_SIZE}')
    logger.log(f'ARCHIVE_SIZE                : {ARCHIVE_SIZE}')
    logger.log(f'NUM_OBJECTIVES              : {NUM_OBJECTIVES}')
    logger.log(f'OPTIMIZATION_DIRECTIONS     : {OPTIMIZATION_DIRECTIONS}')
    logger.log(f'MUTATION_PROBABILITY        : {MUTATION_PROBABILITY}')
    logger.log(f'USE_MULTIPROCESSING         : {USE_MULTIPROCESSING}')
    logger.log(f'DISCARD_INDIVIDUALS         : {DISCARD_INDIVIDUALS}')
    logger.log(f'MUTATION_NUMBER             : {MUTATION_NUMBER}')
    logger.log(f'UNIQUE                      : {UNIQUE}')
    logger.log(f'LSTM_RANDOM                 : {LSTM_RANDOM}')
    logger.log(f'MAX_LSTM_LAYERS             : {MAX_LSTM_LAYERS}')
    logger.log(f'CHANGE_RATE                 : {CHANGE_RATE}')
    logger.log(f'LSTM_UNITS                  : {LSTM_UNITS}')
    logger.log(f'LSTM_ACTIVATION             : {LSTM_ACTIVATION}')
    logger.log(f'LSTM_REC_ACTIVATION         : {LSTM_REC_ACTIVATION}')
    logger.log(f'DENSE_ACTIVATION            : {DENSE_ACTIVATION}')
    logger.log(f'DROPOUT                     : {DROPOUT}')
    logger.log(f'BATCH_NORM                  : {BATCH_NORM}')
    logger.log(f'TRAIN_EPOCHS                : {TRAIN_EPOCHS}')
    logger.log(f'BATCH_SIZE                  : {BATCH_SIZE}')
    logger.log(f'LSTM1_ADD_PROB              : {LSTM1_ADD_PROB}')
    logger.log(f'LSTM1_REMOVE_PROB           : {LSTM1_REMOVE_PROB}')
    logger.log(f'LSTM2_ADD_PROB              : {LSTM2_ADD_PROB}')
    logger.log(f'LSTM2_REMOVE_PROB           : {LSTM2_REMOVE_PROB}')
    logger.log(f'DENSE_ACT_PROB              : {DENSE_ACT_PROB}')
    logger.log(f'LSTM_UNITS_PROB             : {LSTM_UNITS_PROB}')
    logger.log(f'LSTM_ACT_PROB               : {LSTM_ACT_PROB}')
    logger.log(f'LSTM_REC_ACT_PROB           : {LSTM_REC_ACT_PROB}')
    logger.log(f'LSTM_DROPOUT_PROB           : {LSTM_DROPOUT_PROB}')
    logger.log(f'LSTM_BATCHNORM_PROB         : {LSTM_BATCHNORM_PROB}')


def main():
    logger = LSTMAutoencoderLogger()
    log_all_parameters(logger)

    csv_file = 'data/dataset.csv'
    x_train, y_train, x_test, y_test = get_training_testing_windows(csv_file, sequence_length=200, train_size=200)

    # Training & testing parameters
    train_test_data = {
        'epochs': TRAIN_EPOCHS,
        'batch_size': BATCH_SIZE,
        'x_train': x_train,
        'y_train': y_train,
        'x_test': x_test,
        'y_test': y_test
    }

    # Same way we can use crossover parameters.
    mutation_parameters = {
        'mutation_number': MUTATION_NUMBER,
        'unique': UNIQUE,
        'lstm_random': LSTM_RANDOM,
        'max_lstm_layers': MAX_LSTM_LAYERS,
        'change_rate': CHANGE_RATE
    }

    # LSTM Autoencoder parameters
    autoencoder_parameters = {
        'lstm_units': LSTM_UNITS,
        'lstm_activation': LSTM_ACTIVATION,
        'lstm_rec_activation': LSTM_REC_ACTIVATION,
        'dense_activation': DENSE_ACTIVATION,
        'dropout': DROPOUT,
        'batch_norm': BATCH_NORM
    }

    # Probabilities should be sum equal 1.0
    mutation_function = [
        ('LSTM1_ADD',      LSTM1_ADD_PROB),
        ('LSTM1_REMOVE',   LSTM1_REMOVE_PROB),
        ('LSTM2_ADD',      LSTM2_ADD_PROB),
        ('LSTM2_REMOVE',   LSTM2_REMOVE_PROB),
        ('DENSE_ACT',      DENSE_ACT_PROB),
        ('LSTM_UNITS',     LSTM_UNITS_PROB),
        ('LSTM_ACT',       LSTM_ACT_PROB),
        ('LSTM_REC_ACT',   LSTM_REC_ACT_PROB),
        ('LSTM_DROPOUT',   LSTM_DROPOUT_PROB),
        ('LSTM_BATCHNORM', LSTM_BATCHNORM_PROB)]

    # NSGA2 parameters
    nsga2_parameters = {
        'population_size': POPULATION_SIZE,
        'archive_size': ARCHIVE_SIZE,
        'max_generations': MAX_GENERATIONS,
        'num_objectives': NUM_OBJECTIVES,
        'optimization_directions': OPTIMIZATION_DIRECTIONS,
        'mutation_probability': MUTATION_PROBABILITY,
        'use_multiprocessing': USE_MULTIPROCESSING,
        'discard_individuals': DISCARD_INDIVIDUALS
    }

    representation_object = LSTMAutoencoder(input_shape=(180, 6), output_shape=(20, 6),
                                            ae_params=autoencoder_parameters, train_test_data=train_test_data,
                                            mutation_methods=mutation_function, mutation_params=mutation_parameters,
                                            logger=logger)

    ea = NSGA2(nsga2_parameters=nsga2_parameters, representation_object=representation_object, logger=logger)

    ea.run()


if __name__ == '__main__':

    # Nromal = 2
    # GPU = 12
    # TPU = 40
    main()




