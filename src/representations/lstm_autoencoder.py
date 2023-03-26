"""
This class implements the representation of an LSTM autoencoder so that it can be used in evolutionary algorithms.
"""
import uuid
import random
import tensorflow as tf
from memory_profiler import memory_usage
from keras.optimizers import RMSprop
from keras.losses import MeanSquaredError
from src.representations.representation import Representation
from keras.layers import LSTM, Dropout, BatchNormalization, TimeDistributed, Dense, RepeatVector


class LSTMAutoencoder(Representation):
    def __init__(self, input_shape, output_shape, max_lstm_layers, train_test_data, mutation_parameters):
        """
        Create dictionary representation class for LSTM AutoEncoder used for multivariate signal forecasting.
        Whole idea is to forecast 20 new timestamps/rows from initial 180 timestamps/rows of multivariate signal.
        Between encoder and decoder part is RepeatVector(20) layer for multistep forcasting.

        Example of final representation:

        {
            "num_encoder_layers": 2,
            "num_decoder_layers": 2,
            "dense_activation": "linear",
            "encoder": [
                {
                    "type": "LSTM",
                    "units": 64,
                    "activation": "relu",
                    "recurrent_activation": "sigmoid",
                    "dropout": false,
                    "use_batchnorm": false
                },
                {
                    "type": "LSTM",
                    "units": 32,
                    "activation": "relu",
                    "recurrent_activation": "hard_sigmoid",
                    "dropout": 0.2,
                    "use_batchnorm": true
                }
            ],
            "decoder": [
                {
                    "type": "LSTM",
                    "units": 32,
                    "activation": "relu",
                    "recurrent_activation": "sigmoid",
                    "dropout": false,
                    "use_batchnorm": false
                },
                {
                    "type": "LSTM",
                    "units": 64,
                    "activation": "relu",
                    "recurrent_activation": "hard_sigmoid",
                    "dropout": 0.2,
                    "use_batchnorm": true
                }
            ]
        }

         If dropout is false, there is no Dropout layer. If dropout is a number in the range 0 to 1, a Dropout layer
         is added with the specified dropout rate. The use_batchnorm parameter is used to indicate whether a
         BatchNormalization layer should be added after LSTM layer.

        :param input_shape: Input shape e.g. (180, 6), 180 rows with 6 values.
        :param output_shape: Number of timestamps and number of forecasted values in one timestamp, e.g. (20, 6).
        :param max_lstm_layers: Defines maximal number of lstm layers in encoder and decoder, used in mutation.
        :param train_test_data: Is dictionary with x_train, y_train, x_test and y_test numpy arrays, those arrays must.
        be in specific shape for LSTM Autoencoder, e.g. (samples, 180, 6) for x_train and (samples, 20, 6) for y_train.
        Function for creating such dictionary is in src/utils/dataset_tools.py.
        :param mutation_parameters: Parameters for method mutate of this class.
        """
        # Unique name of initialized object.
        self.__name__ = str(uuid.uuid4())[:8]
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.max_lstm_layers = max_lstm_layers
        self.train_test_data = train_test_data
        self.mutation_parameters = mutation_parameters

        self.num_encoder_layers = None
        self.num_decoder_layers = None
        self.dense_activation = None
        self.encoder_layers = []
        self.decoder_layers = []

        self.model = None

    def _create_initial_representation(self):
        """
        Creates basic LSTM Neural Network with just one LSTM layer in encoder and decoder.
        In the model we want to forecast e.g. from 180 timestamps/rows next 20, so we need to use
        return sequence parameter of LSTM layer equal True.
        Only last LSTM layer of encoder has return sequence equal False.
        """
        self.num_encoder_layers = 1
        self.num_decoder_layers = 1
        self.dense_activation = 'linear'
        self.encoder_layers = [
            {
                "type": "LSTM",
                "units": 16,
                "activation": "tanh",
                "recurrent_activation": "sigmoid",
                "dropout": 0.2,
                "use_batchnorm": True
            }
        ]

        self.decoder_layers = [
            {
                "type": "LSTM",
                "units": 16,
                "activation": "tanh",
                "recurrent_activation": "sigmoid",
                "dropout": 0.2,
                "use_batchnorm": True
            }
        ]

    def encode(self):
        """
        Create encoded version of model architecture.

        :return: Architecture dictionary.
        """
        encoded_architecture = {
            "num_encoder_layers": self.num_encoder_layers,
            "num_decoder_layers": self.num_decoder_layers,
            "dense_activation": self.dense_activation,
            "encoder_layers": self.encoder_layers,
            "decoder_layers": self.decoder_layers
        }
        return encoded_architecture

    def decode(self, encoded_representation):
        """
        Load encoded architecture into inner representation in class (update class variables).

        :param encoded_representation: Representation created by method encode or manually.
        """
        self.num_encoder_layers = int(encoded_representation["num_encoder_layers"])
        self.num_decoder_layers = int(encoded_representation["num_decoder_layers"])
        self.dense_activation = encoded_representation["dense_activation"]
        self.encoder_layers = encoded_representation["encoder_layers"]
        self.decoder_layers = encoded_representation["decoder_layers"]

    def display_representation(self):
        """
        Display representation of architecture.
        """
        print("")
        print("Input shape: ", self.input_shape)
        print("Output shape: ", self.output_shape)
        print("Number of encoder LSTM layers:", self.num_encoder_layers)
        print("Number of decoder LSTM layers:", self.num_decoder_layers)
        print("Activation function of Dense layer:", self.dense_activation)
        print("Architecture:")
        print("Encoder:")
        for layer in self.encoder_layers:
            print(layer)
        print("Decoder:")
        for layer in self.decoder_layers:
            print(layer)
        print("")

    def _build_model(self):
        """
        Create Tensorflow model from representation.
        When we are forecasting multivariate signal multiple steps into future
        e.g. from 180 samples forecast 20 steps in future last layer is TimeDistributed(Dense()) layer
        and between encoder and decoder is RepeatVector(20) layer.
        As optimizer is used RMSprop commonly used in recurrent nn and as loss function Mean Squared Error.
        Adam and MAE is also possible to use.

        Example summary:
            ___________________________________________________________________________________
             Layer (type)                                Output Shape              Param #
            ===================================================================================
             input_1 (InputLayer)                        [(None, 180, 6)]          0

             lstm (LSTM)                                 (None, 64)                18176

             dropout (Dropout)                           (None, 64)                0

             batch_normalization (BatchNormalization)    (None, 64)                256

             repeat_vector (RepeatVector)                (None, 20, 64)            0

             lstm_1 (LSTM)                               (None, 20, 64)            33024

             dropout_1 (Dropout)                         (None, 20, 64)            0

             batch_normalization_1 (BatchNormalization)  (None, 20, 64)            256

             time_distributed (TimeDistributed)          (None, 20, 6)             390

            =====================================================================================

        :return: compiled keras model
        """
        inputs = tf.keras.Input(shape=self.input_shape)
        arch = inputs
        encoded_architecture = self.encode()

        # Create encoder part
        for i, layer_params in enumerate(encoded_architecture['encoder_layers']):
            # Last LSTM layer has return_sequence=False
            if i == (len(encoded_architecture['encoder_layers']) - 1):
                arch = LSTM(layer_params['units'], activation=layer_params['activation'],
                            recurrent_activation=layer_params['recurrent_activation'], return_sequences=False)(arch)

            # Every other LSTM layer has return_sequence=True
            else:
                arch = LSTM(layer_params['units'], activation=layer_params['activation'],
                            recurrent_activation=layer_params['recurrent_activation'], return_sequences=True)(arch)

            # Dropout layer
            if layer_params["dropout"]:
                arch = Dropout(layer_params["dropout"])(arch)

            # BatchNormalization layer
            if layer_params["use_batchnorm"]:
                arch = BatchNormalization()(arch)

        # Add middle layer RepeatVector into architecture model, self.output_shape[0]
        # determines number of forecasted time steps
        arch = RepeatVector(self.output_shape[0])(arch)

        # Create decoder part
        for i, layer_params in enumerate(encoded_architecture['decoder_layers']):
            arch = LSTM(layer_params['units'], activation=layer_params['activation'],
                        recurrent_activation=layer_params['recurrent_activation'], return_sequences=True)(arch)

            # Dropout layer
            if layer_params["dropout"]:
                arch = Dropout(layer_params["dropout"])(arch)

            # BatchNormalization layer
            if layer_params["use_batchnorm"]:
                arch = BatchNormalization()(arch)

        # Add last layer (Dense), which is TimeDistributed
        arch = TimeDistributed(Dense(self.output_shape[1], activation=self.dense_activation))(arch)

        # Create Tensorflow model
        self.model = tf.keras.Model(inputs=inputs, outputs=arch)
        self.model.compile(optimizer=RMSprop(learning_rate=0.001),
                           loss=MeanSquaredError(),
                           metrics=['mse', 'mape'])
        return self.model

    def _train_model(self):
        """
        Train Model with parameters saved in self.train_test_data.

        :returns: Trained model.
        """
        model = self._build_model()
        epochs = self.train_test_data['epochs']
        batch_size = self.train_test_data['batch_size']
        x_train = self.train_test_data['x_train']
        y_train = self.train_test_data['y_train']
        x_test = self.train_test_data['x_test']
        y_test = self.train_test_data['y_test']

        model.fit(x_train, y_train, epochs=epochs,
                  batch_size=batch_size,
                  validation_data=(x_test, y_test),
                  shuffle=False,
                  verbose=False)

        self.model = model
        return self.model

    def _add_lstm_mutation(self, lstm_random):
        """
        Adds one more LSTM layer on random space of architecture if maximal number of layers is not reached.

        :param lstm_random: True/False -> whether to randomly set hyper-parameters of new LST layer
        or copy hyper-parameters of last LSTM layer.
        :return: Mutated LSTMAutoencoder object or False if number of maximal number of layers has been reached.
        """
        # Random possibilities of hyper-parameters values
        units = [8, 16, 32, 64, 128, 256]
        activation = ['sigmoid', 'tanh', 'relu', 'softmax', 'linear']
        rec_activation = ['sigmoid', 'hard_sigmoid', 'tanh', 'relu', 'linear']
        dropout_choices = [False, 0.1, 0.2, 0.3, 0.4, 0.5]
        batch_norm_choices = [True, False]

        # Choose whether to mutate encoder od decoder part, encoder -> 0, decoder -> 1
        architecture_part = random.choice([('num_encoder_layers', 'encoder_layers'),
                                           ('num_decoder_layers', 'decoder_layers')])

        encoded_architecture = self.encode()
        if int(encoded_architecture[architecture_part[0]]) < self.max_lstm_layers:
            encoded_architecture[architecture_part[0]] = str(int(encoded_architecture[architecture_part[0]]) + 1)

            # Randomly choose hyper-parameters of new LSTM layer
            if lstm_random:
                lstm_layers = encoded_architecture[architecture_part[1]]

                new_lstm_layer = {
                    "type": "LSTM",
                    "units": random.choice(units),
                    "activation": random.choice(activation),
                    "recurrent_activation": random.choice(rec_activation),
                    "dropout": random.choice(dropout_choices),
                    "use_batchnorm": random.choice(batch_norm_choices)
                }

                random_index = random.randint(0, len(lstm_layers))
                lstm_layers.insert(random_index, new_lstm_layer)
                encoded_architecture[architecture_part[1]] = lstm_layers
                mutated_lstm_autoencoder = LSTMAutoencoder(self.input_shape,
                                                           self.output_shape,
                                                           self.max_lstm_layers,
                                                           self.train_test_data,
                                                           self.mutation_parameters)
                mutated_lstm_autoencoder.decode(encoded_architecture)
                return mutated_lstm_autoencoder

            # Copy last lstm layer and add it into architecture
            else:
                lstm_layers = encoded_architecture[architecture_part[1]]
                # Get last LSTM layer
                new_lstm_layer = lstm_layers[-1]
                random_index = random.randint(0, len(lstm_layers))
                lstm_layers.insert(random_index, new_lstm_layer)
                encoded_architecture[architecture_part[1]] = lstm_layers
                mutated_lstm_autoencoder = LSTMAutoencoder(self.input_shape,
                                                           self.output_shape,
                                                           self.max_lstm_layers,
                                                           self.train_test_data,
                                                           self.mutation_parameters)
                mutated_lstm_autoencoder.decode(encoded_architecture)
                return mutated_lstm_autoencoder
        else:
            # If mutation is not possible (max lstm layers) return False and do another mutation
            return False

    def _remove_lstm_mutation(self):
        """
        Remove random LSTM layer from architecture.

        :return: Mutated LSTMAutoencoder object or False if number of maximal number of layers has been reached.
        """
        # Choose whether to mutate encoder od decoder part, encoder -> 0, decoder -> 1
        architecture_part = random.choice([('num_encoder_layers', 'encoder_layers'),
                                           ('num_decoder_layers', 'decoder_layers')])

        encoded_architecture = self.encode()
        if int(encoded_architecture[architecture_part[0]]) > 1:
            encoded_architecture[architecture_part[0]] = str(int(encoded_architecture[architecture_part[0]]) - 1)
            lstm_layers = encoded_architecture[architecture_part[1]]
            # Remove random LSTM layer from architecture
            index = random.randint(0, len(lstm_layers) - 1)
            lstm_layers.pop(index)
            encoded_architecture[architecture_part[1]] = lstm_layers
            mutated_lstm_autoencoder = LSTMAutoencoder(self.input_shape,
                                                       self.output_shape,
                                                       self.max_lstm_layers,
                                                       self.train_test_data,
                                                       self.mutation_parameters)
            mutated_lstm_autoencoder.decode(encoded_architecture)
            return mutated_lstm_autoencoder
        else:
            # If mutation is not possible return False and do another mutation
            return False

    def _dense_layer_mutation(self):
        """
        Mutate activation function of last Dense layer.

        :return: Mutated LSTMAutoencoder object.
        """
        encoded_architecture = self.encode()
        dense_activation = encoded_architecture['dense_activation']
        dense_activation_functions = ['sigmoid', 'tanh', 'softmax', 'linear']
        # Remove used activation function to avoid repeating/neutral mutation
        dense_activation_functions.remove(dense_activation)
        encoded_architecture['dense_activation'] = random.choice(dense_activation_functions)
        mutated_lstm_autoencoder = LSTMAutoencoder(self.input_shape,
                                                   self.output_shape,
                                                   self.max_lstm_layers,
                                                   self.train_test_data,
                                                   self.mutation_parameters)
        mutated_lstm_autoencoder.decode(encoded_architecture)
        return mutated_lstm_autoencoder

    def _lstm_units_mutation(self, change_rate):
        """
        Mutate number of LSTM units in random LSTM layer.

        :param change_rate: Rate of units change, e.g. 0.2 mean range -20%/+20% of previous value.
        :return: Mutated LSTMAutoencoder object.
        """
        # Choose whether to mutate encoder od decoder part, encoder -> 0, decoder -> 1
        architecture_part = random.choice([('num_encoder_layers', 'encoder_layers'),
                                           ('num_decoder_layers', 'decoder_layers')])

        encoded_architecture = self.encode()
        lstm_layers = encoded_architecture[architecture_part[1]]
        random_lstm_layer_idx = random.randint(0, len(lstm_layers)-1)
        random_lstm_layer = lstm_layers[random_lstm_layer_idx]
        current_lstm_units = random_lstm_layer['units']
        lower_bound = int(current_lstm_units * (1 - change_rate))
        upper_bound = int(current_lstm_units * (1 + change_rate))
        new_lstm_units = random.randint(lower_bound, upper_bound)
        lstm_layers[random_lstm_layer_idx]['units'] = new_lstm_units
        encoded_architecture[architecture_part[1]] = lstm_layers
        mutated_lstm_autoencoder = LSTMAutoencoder(self.input_shape,
                                                   self.output_shape,
                                                   self.max_lstm_layers,
                                                   self.train_test_data,
                                                   self.mutation_parameters)
        mutated_lstm_autoencoder.decode(encoded_architecture)
        return mutated_lstm_autoencoder

    def _lstm_activation_mutation(self):
        """
        Mutate activation function in random LSTM layer.

        :return: Mutated LSTMAutoencoder object.
        """
        # Choose whether to mutate encoder od decoder part, encoder -> 0, decoder -> 1
        architecture_part = random.choice([('num_encoder_layers', 'encoder_layers'),
                                           ('num_decoder_layers', 'decoder_layers')])

        encoded_architecture = self.encode()
        lstm_layers = encoded_architecture[architecture_part[1]]
        random_lstm_layer_idx = random.randint(0, len(lstm_layers) - 1)
        random_lstm_layer = lstm_layers[random_lstm_layer_idx]
        current_lstm_activation = random_lstm_layer['activation']
        activation = ['sigmoid', 'tanh', 'relu', 'softmax', 'linear']
        activation.remove(current_lstm_activation)
        lstm_layers[random_lstm_layer_idx]['activation'] = random.choice(activation)
        encoded_architecture[architecture_part[1]] = lstm_layers
        mutated_lstm_autoencoder = LSTMAutoencoder(self.input_shape,
                                                   self.output_shape,
                                                   self.max_lstm_layers,
                                                   self.train_test_data,
                                                   self.mutation_parameters)
        mutated_lstm_autoencoder.decode(encoded_architecture)
        return mutated_lstm_autoencoder

    def _lstm_rec_activation_mutation(self):
        """
        Mutate recurrent activation function in random LSTM layer.

        :return: Mutated LSTMAutoencoder object.
        """
        # Choose whether to mutate encoder od decoder part, encoder -> 0, decoder -> 1
        architecture_part = random.choice([('num_encoder_layers', 'encoder_layers'),
                                           ('num_decoder_layers', 'decoder_layers')])

        encoded_architecture = self.encode()
        lstm_layers = encoded_architecture[architecture_part[1]]
        random_lstm_layer_idx = random.randint(0, len(lstm_layers) - 1)
        random_lstm_layer = lstm_layers[random_lstm_layer_idx]
        current_lstm_rec_activation = random_lstm_layer['recurrent_activation']
        rec_activation = ['sigmoid', 'hard_sigmoid', 'tanh', 'relu', 'linear']
        rec_activation.remove(current_lstm_rec_activation)
        lstm_layers[random_lstm_layer_idx]['recurrent_activation'] = random.choice(rec_activation)
        encoded_architecture[architecture_part[1]] = lstm_layers
        mutated_lstm_autoencoder = LSTMAutoencoder(self.input_shape,
                                                   self.output_shape,
                                                   self.max_lstm_layers,
                                                   self.train_test_data,
                                                   self.mutation_parameters)
        mutated_lstm_autoencoder.decode(encoded_architecture)
        return mutated_lstm_autoencoder

    def _lstm_dropout_mutation(self):
        """
        Mutate dropout parameter in random LSTM layer.

        :return: Mutated LSTMAutoencoder object.
        """
        # Choose whether to mutate encoder od decoder part, encoder -> 0, decoder -> 1
        architecture_part = random.choice([('num_encoder_layers', 'encoder_layers'),
                                           ('num_decoder_layers', 'decoder_layers')])

        encoded_architecture = self.encode()
        lstm_layers = encoded_architecture[architecture_part[1]]
        random_lstm_layer_idx = random.randint(0, len(lstm_layers) - 1)
        random_lstm_layer = lstm_layers[random_lstm_layer_idx]
        current_lstm_dropout = random_lstm_layer['dropout']
        dropouts = [False, 0.1, 0.2, 0.3, 0.4, 0.5]
        dropouts.remove(current_lstm_dropout)
        lstm_layers[random_lstm_layer_idx]['dropout'] = random.choice(dropouts)
        encoded_architecture[architecture_part[1]] = lstm_layers
        mutated_lstm_autoencoder = LSTMAutoencoder(self.input_shape,
                                                   self.output_shape,
                                                   self.max_lstm_layers,
                                                   self.train_test_data,
                                                   self.mutation_parameters)
        mutated_lstm_autoencoder.decode(encoded_architecture)
        return mutated_lstm_autoencoder

    def _lstm_batch_norm_mutation(self):
        """
        Mutate use_batchnorm parameter in random LSTM layer.

        :return: Mutated LSTMAutoencoder object.
        """
        # Choose whether to mutate encoder od decoder part, encoder -> 0, decoder -> 1
        architecture_part = random.choice([('num_encoder_layers', 'encoder_layers'),
                                           ('num_decoder_layers', 'decoder_layers')])

        encoded_architecture = self.encode()
        lstm_layers = encoded_architecture[architecture_part[1]]
        random_lstm_layer_idx = random.randint(0, len(lstm_layers) - 1)
        random_lstm_layer = lstm_layers[random_lstm_layer_idx]
        current_lstm_use_batch = random_lstm_layer['use_batchnorm']
        batch_norm_uses = [True, False]
        batch_norm_uses.remove(current_lstm_use_batch)
        lstm_layers[random_lstm_layer_idx]['use_batchnorm'] = random.choice(batch_norm_uses)
        encoded_architecture[architecture_part[1]] = lstm_layers
        mutated_lstm_autoencoder = LSTMAutoencoder(self.input_shape,
                                                   self.output_shape,
                                                   self.max_lstm_layers,
                                                   self.train_test_data,
                                                   self.mutation_parameters)
        mutated_lstm_autoencoder.decode(encoded_architecture)
        return mutated_lstm_autoencoder

    def mutate(self):
        """
        Mutate current LSTM Autoencoder and returns new mutated object.
        This method always do some mutation, it means in case of failed mutation method it will randomly choose
        from method which will not fail.

        # TODO choose/remove which method to use or not -> parameter[list] which will update mutation_methods list.
        """
        mutation_number = self.mutation_parameters['mutation_number']
        unique = self.mutation_parameters['unique']
        lstm_random = self.mutation_parameters['lstm_random']
        change_rate = self.mutation_parameters['change_rate']

        # Mutation methods in form: (method, parameters dictionary)
        mutation_methods = [(self._add_lstm_mutation, {'hyper_random': lstm_random}),
                            (self._remove_lstm_mutation, {}),
                            (self._dense_layer_mutation, {}),
                            (self._lstm_units_mutation, {'change_rate': change_rate}),
                            (self._lstm_activation_mutation, {}),
                            (self._lstm_rec_activation_mutation, {}),
                            (self._lstm_dropout_mutation, {}),
                            (self._lstm_batch_norm_mutation, {})]

        mutated_lstm_autoencoder = None

        # Choose only unique mutations
        if unique:
            mutations = random.sample(mutation_methods, k=mutation_number)
            for mutation in mutations:
                method = mutation[0]
                params = mutation[1]
                mutated_lstm_autoencoder = method(**params)

                # If any mutations fails, randomly choose another one
                if not mutated_lstm_autoencoder:
                    nonfail_mutation_methods = [(self._dense_layer_mutation, {}),
                                                (self._lstm_units_mutation, {'change_rate': change_rate}),
                                                (self._lstm_activation_mutation, {}),
                                                (self._lstm_rec_activation_mutation, {}),
                                                (self._lstm_dropout_mutation, {}),
                                                (self._lstm_batch_norm_mutation, {})]
                    method, params = random.choice(nonfail_mutation_methods)
                    mutated_lstm_autoencoder = method(**params)

                self.decode(mutated_lstm_autoencoder.encode())

        # Mutations can repeat
        else:
            mutations = random.choices(mutation_methods, k=mutation_number)
            for mutation in mutations:
                method = mutation[0]
                params = mutation[1]
                mutated_lstm_autoencoder = method(**params)

                # If any mutations fails, randomly choose another one
                if not mutated_lstm_autoencoder:
                    nonfail_mutation_methods = [(self._dense_layer_mutation, {}),
                                                (self._lstm_units_mutation, {'change_rate': change_rate}),
                                                (self._lstm_activation_mutation, {}),
                                                (self._lstm_rec_activation_mutation, {}),
                                                (self._lstm_dropout_mutation, {}),
                                                (self._lstm_batch_norm_mutation, {})]
                    method, params = random.choice(nonfail_mutation_methods)
                    mutated_lstm_autoencoder = method(**params)

                self.decode(mutated_lstm_autoencoder.encode())

        return mutated_lstm_autoencoder

    def crossover(self, other_representation):
        """
        One-point crossover of two architectures. Crossover point is chosen randomly.

        :param other_representation: Other LSTMAutoencoder object.
        :return: Two crossed LSTMAutoencoder objects.
        """
        self_encoded = self.encode()
        other_encoded = other_representation.encode()

        # 1. encoder and 2. decoder
        self_encoded['num_decoder_layers'] = other_encoded['num_decoder_layers']
        self_encoded['decoder_layers'] = other_encoded['decoder_layers']

        # 2. encoder and 1.encoder
        other_encoded['num_encoder_layers'] = self_encoded['num_encoder_layers']
        other_encoded['encoder_layers'] = self_encoded['encoder_layers']

        crossed1_lstm_ae = LSTMAutoencoder(self.input_shape,
                                           self.output_shape,
                                           self.max_lstm_layers,
                                           self.train_test_data,
                                           self.mutation_parameters)
        crossed1_lstm_ae.decode(self_encoded)

        crossed2_lstm_ae = LSTMAutoencoder(self.input_shape,
                                           self.output_shape,
                                           self.max_lstm_layers,
                                           self.train_test_data,
                                           self.mutation_parameters)
        crossed2_lstm_ae.decode(other_encoded)

        return crossed1_lstm_ae, crossed2_lstm_ae

    def create_initial_population(self, population_size):
        """
        Create initial population for evolutionary algorithm.
        New individuals are created using random 5 mutations.

        :param population_size: Size of population.
        :return: Initialized population.
        """
        initial_population = []
        for i in range(population_size):
            self._create_initial_representation()
            mutated_individual = self.mutate()
            initial_population.append(mutated_individual)

        return initial_population

    def evaluate(self):
        """
        Evaluate LSTM Autoencoder.
        If the model is not functional, the mutation will be used until the model is functional.
        mape - mean absolute percentage error
        num_params - number of model parameters
        memory_usage_value - bytes used for predicting one value

        :return: [mape, num_params, memory_usage_value]
        """
        self._build_model()
        self._train_model()

        x_test = self.train_test_data['x_test']
        y_test = self.train_test_data['y_test']

        # Accuracy of forecasting
        _, mse, mape = self.model.evaluate(x_test, y_test, verbose=False)

        # Computational complexity
        num_params = self.model.count_params()

        # Memory usage
        memory_usage_tuple = memory_usage((self.model.predict, (x_test,), {'batch_size': 1}))
        memory_usage_value = max(memory_usage_tuple) - min(memory_usage_tuple)

        result = [mape, num_params, memory_usage_value]

        if str(mape) == 'nan':
            self.mutate()
            result = self.evaluate()

        return result

    def get_id(self):
        """
        Name getter.

        :return: __name__
        """
        return self.__name__
