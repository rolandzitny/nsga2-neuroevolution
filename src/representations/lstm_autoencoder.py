"""
This class implements the representation of an LSTM autoencoder so that it can be used in evolutionary algorithms.
"""
import math
import uuid
import random
import tensorflow as tf
import tracemalloc
from keras.optimizers import RMSprop
from keras.losses import MeanSquaredError
from src.representations.representation import Representation
from keras.layers import LSTM, Dropout, BatchNormalization, TimeDistributed, Dense, RepeatVector


class LSTMAutoencoder(Representation):
    def __init__(self, input_shape, output_shape, train_test_data, mutation_methods, mutation_parameters):
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
        :param train_test_data: Is dictionary with x_train, y_train, x_test and y_test numpy arrays, those arrays must.
        be in specific shape for LSTM Autoencoder, e.g. (samples, 180, 6) for x_train and (samples, 20, 6) for y_train.
        Function for creating such dictionary is in src/utils/dataset_tools.py.
        :param mutation_parameters: Parameters for method mutate of this class.
        """
        # Unique name of initialized object.
        self.__name__ = str(uuid.uuid4())[:8]
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.train_test_data = train_test_data
        self.mutation_methods = mutation_methods
        self.mutation_parameters = mutation_parameters

        self.num_encoder_layers = None
        self.num_decoder_layers = None
        self.dense_activation = None
        self.encoder_layers = []
        self.decoder_layers = []
        self.model = None

        self.mutation_number = self.mutation_parameters['mutation_number']
        self.unique = self.mutation_parameters['unique']
        lstm_random = self.mutation_parameters['lstm_random']
        max_lstm_layers = self.mutation_parameters['max_lstm_layers']
        change_rate = self.mutation_parameters['change_rate']

        self.prepared_mutation_methods = []
        self.prepared_mutation_methods_nonfail = []

        # Mutation methods in form: (method, parameters dictionary)
        for mutation_name in self.mutation_methods:
            if mutation_name == 'LSTM1_ADD':
                method = (self._add_lstm1_mutation, {'lstm_random': lstm_random, 'max_lstm_layers': max_lstm_layers})
                self.prepared_mutation_methods.append(method)

            elif mutation_name == 'LSTM1_REMOVE':
                method = (self._remove_lstm1_mutation, {})
                self.prepared_mutation_methods.append(method)

            elif mutation_name == 'LSTM2_ADD':
                method = (self._add_lstm2_mutation, {'lstm_random': lstm_random, 'max_lstm_layers': max_lstm_layers})
                self.prepared_mutation_methods.append(method)

            elif mutation_name == 'LSTM2_REMOVE':
                method = (self._remove_lstm2_mutation, {})
                self.prepared_mutation_methods.append(method)

            elif mutation_name == 'DENSE_ACT':
                method = (self._dense_layer_mutation, {})
                self.prepared_mutation_methods.append(method)
                self.prepared_mutation_methods_nonfail.append(method)

            elif mutation_name == 'LSTM_UNITS':
                method = (self._lstm_units_mutation, {'change_rate': change_rate})
                self.prepared_mutation_methods.append(method)
                self.prepared_mutation_methods_nonfail.append(method)

            elif mutation_name == 'LSTM_ACT':
                method = (self._lstm_activation_mutation, {})
                self.prepared_mutation_methods.append(method)
                self.prepared_mutation_methods_nonfail.append(method)

            elif mutation_name == 'LSTM_REC_ACT':
                method = (self._lstm_rec_activation_mutation, {})
                self.prepared_mutation_methods.append(method)
                self.prepared_mutation_methods_nonfail.append(method)

            elif mutation_name == 'LSTM_DROPOUT':
                method = (self._lstm_dropout_mutation, {})
                self.prepared_mutation_methods.append(method)
                self.prepared_mutation_methods_nonfail.append(method)

            elif mutation_name == 'LSTM_BATCHNORM':
                method = (self._lstm_batch_norm_mutation, {})
                self.prepared_mutation_methods.append(method)
                self.prepared_mutation_methods_nonfail.append(method)

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
                "units": 128,
                "activation": "tanh",
                "recurrent_activation": "sigmoid",
                "dropout": 0.2,
                "use_batchnorm": True
            },
            {
                "type": "LSTM",
                "units": 64,
                "activation": "tanh",
                "recurrent_activation": "sigmoid",
                "dropout": 0.2,
                "use_batchnorm": True
            }
        ]

        self.decoder_layers = [
            {
                "type": "LSTM",
                "units": 64,
                "activation": "tanh",
                "recurrent_activation": "sigmoid",
                "dropout": 0.2,
                "use_batchnorm": True
            },
            {
                "type": "LSTM",
                "units": 128,
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
                  verbose=0)

        self.model = model
        return self.model

    def _add_lstm2_mutation(self, lstm_random, max_lstm_layers):
        """
        Adds one more LSTM layer at end of encoder part and one at start of decoder part
        if maximal number of layers is not reached.

        :param lstm_random: True/False -> whether to randomly set hyper-parameters of new LST layer
        or copy hyper-parameters of last LSTM layer.
        :param max_lstm_layers: Maximal number of LSTM layers in encoder and decoder parts.
        :return: Mutated encoded architecture.
        """
        units = [16, 32, 64, 128, 256]
        activation = ['sigmoid', 'tanh', 'relu', 'softmax', 'linear']
        rec_activation = ['sigmoid', 'hard_sigmoid', 'tanh', 'relu', 'linear']
        dropout_choices = [False, 0.1, 0.2, 0.3, 0.4, 0.5]
        batch_norm_choices = [True, False]

        encoded_architecture = self.encode()

        encoder_length = int(encoded_architecture['num_encoder_layers'])
        decoder_length = int(encoded_architecture['num_decoder_layers'])
        encoder_layers = encoded_architecture['encoder_layers']
        decoder_layers = encoded_architecture['decoder_layers']

        if encoder_length < max_lstm_layers and decoder_length < max_lstm_layers:
            # Randomly choose hyper-parameters of new LSTM layer
            if lstm_random:
                encoder_lstm_layer = {
                    "type": "LSTM",
                    "units": random.choice(units),
                    "activation": random.choice(activation),
                    "recurrent_activation": random.choice(rec_activation),
                    "dropout": random.choice(dropout_choices),
                    "use_batchnorm": random.choice(batch_norm_choices)
                }

                decoder_lstm_layer = {
                    "type": "LSTM",
                    "units": random.choice(units),
                    "activation": random.choice(activation),
                    "recurrent_activation": random.choice(rec_activation),
                    "dropout": random.choice(dropout_choices),
                    "use_batchnorm": random.choice(batch_norm_choices)
                }

                encoder_layers.insert(encoder_length, encoder_lstm_layer)
                encoded_architecture['encoder_layers'] = encoder_layers
                encoded_architecture['num_encoder_layers'] = str(encoder_length + 1)

                decoder_layers.insert(0, decoder_lstm_layer)
                encoded_architecture['decoder_layers'] = decoder_layers
                encoded_architecture['num_decoder_layers'] = str(decoder_length + 1)

                return encoded_architecture

            # Copy last lstm layer and add it into architecture
            else:
                encoder_lstm_layer = encoder_layers[-1]
                decoder_lstm_layer = decoder_layers[0]

                encoder_layers.insert(encoder_length, encoder_lstm_layer)
                encoded_architecture['encoder_layers'] = encoder_layers
                encoded_architecture['num_encoder_layers'] = str(encoder_length + 1)

                decoder_layers.insert(0, decoder_lstm_layer)
                encoded_architecture['decoder_layers'] = decoder_layers
                encoded_architecture['num_decoder_layers'] = str(decoder_length + 1)

                return encoded_architecture
        else:
            return False

    def _remove_lstm2_mutation(self):
        """
        Remove LSTM layer from end of encoder and start of decoder.

        :return: Mutated encoded architecture.
        """
        encoded_architecture = self.encode()

        encoder_length = int(encoded_architecture['num_encoder_layers'])
        decoder_length = int(encoded_architecture['num_decoder_layers'])
        encoder_layers = encoded_architecture['encoder_layers']
        decoder_layers = encoded_architecture['decoder_layers']

        if encoder_length > 1 and decoder_length > 1:
            encoder_layers.pop(encoder_length - 1)
            decoder_layers.pop(0)

            encoded_architecture['encoder_layers'] = encoder_layers
            encoded_architecture['num_encoder_layers'] = str(encoder_length - 1)

            encoded_architecture['decoder_layers'] = decoder_layers
            encoded_architecture['num_decoder_layers'] = str(decoder_length - 1)

            return encoded_architecture

        else:
            # If mutation is not possible return False and do another mutation
            return False

    def _add_lstm1_mutation(self, lstm_random, max_lstm_layers):
        """
        Adds one more LSTM layer on random space of architecture if maximal number of layers is not reached.

        :param lstm_random: True/False -> whether to randomly set hyper-parameters of new LST layer
        or copy hyper-parameters of last LSTM layer.
        :param max_lstm_layers: Maximal number of LSTM layers in encoder and decoder parts.
        :return: Mutated encoded architecture.
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
        if int(encoded_architecture[architecture_part[0]]) < max_lstm_layers:
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
                return encoded_architecture

            # Copy last lstm layer and add it into architecture
            else:
                lstm_layers = encoded_architecture[architecture_part[1]]
                # Get last LSTM layer
                new_lstm_layer = lstm_layers[-1]
                random_index = random.randint(0, len(lstm_layers))
                lstm_layers.insert(random_index, new_lstm_layer)
                encoded_architecture[architecture_part[1]] = lstm_layers
                return encoded_architecture
        else:
            # If mutation is not possible (max lstm layers) return False and do another mutation
            return False

    def _remove_lstm1_mutation(self):
        """
        Remove random LSTM layer from architecture.

        :return: Mutated encoded architecture.
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

            return encoded_architecture
        else:
            # If mutation is not possible return False and do another mutation
            return False

    def _dense_layer_mutation(self):
        """
        Mutate activation function of last Dense layer.

        :return: Mutated encoded architecture.
        """
        encoded_architecture = self.encode()
        dense_activation = encoded_architecture['dense_activation']
        dense_activation_functions = ['sigmoid', 'tanh', 'softmax', 'linear']
        # Remove used activation function to avoid repeating/neutral mutation
        dense_activation_functions.remove(dense_activation)
        encoded_architecture['dense_activation'] = random.choice(dense_activation_functions)
        return encoded_architecture

    def _lstm_units_mutation(self, change_rate):
        """
        Mutate number of LSTM units in random LSTM layer.

        :param change_rate: Rate of units change, e.g. 0.2 mean range -20%/+20% of previous value.
        :return: Mutated encoded architecture.
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
        return encoded_architecture

    def _lstm_activation_mutation(self):
        """
        Mutate activation function in random LSTM layer.

        :return: Mutated encoded architecture.
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
        return encoded_architecture

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
        return encoded_architecture

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
        return encoded_architecture

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
        return encoded_architecture

    def mutate(self):
        """
        Mutate current LSTM Autoencoder and returns new mutated object.
        This method always do some mutation, it means in case of failed mutation method it will randomly choose
        from method which will not fail.

        :mutation_parameters mutation_number: Number of mutations.
        :mutation_parameters unique: True/False -> When mutation_number is bigger than,
        whether to use only unique mutations.
        :mutation_parameters lstm_random: True/False -> Whether to randomly set hyper-parameters of new LSTM layer.
        :mutation_parameters max_lstm_layers: Defines maximal number of lstm layers in encoder and decoder parts.
        :mutation_parameters change_rate: Rate of units change, for example 0.2.
        """
        mutated_architecture = self.encode()
        # Choose only unique mutations
        if self.unique:
            mutations = random.sample(self.prepared_mutation_methods, k=self.mutation_number)
            for mutation in mutations:
                method = mutation[0]
                params = mutation[1]
                mutated_architecture = method(**params)

                # If any mutations fails, randomly choose another one
                if not mutated_architecture:
                    try:
                        method, params = random.choice(self.prepared_mutation_methods_nonfail)
                        mutated_architecture = method(**params)
                    except:
                        mutated_architecture = self.encode()

                self.decode(mutated_architecture)

        # Mutations can repeat
        else:
            mutations = random.choices(self.prepared_mutation_methods, k=self.mutation_number)
            for mutation in mutations:
                method = mutation[0]
                params = mutation[1]
                mutated_lstm_autoencoder = method(**params)

                # If any mutations fails, randomly choose another one
                if not mutated_lstm_autoencoder:
                    try:
                        method, params = random.choice(self.prepared_mutation_methods_nonfail)
                        mutated_architecture = method(**params)
                    except:
                        mutated_architecture = self.encode()

                self.decode(mutated_architecture)

        return self

    def crossover(self, other_representation):
        """
        One-point crossover of two architectures. Crossover point is chosen randomly.

        :param other_representation: Other LSTMAutoencoder object.
        :return: One crossed LSTMAutoencoder objects.
        """
        # Choose whether to mutate encoder od decoder part, encoder -> 0, decoder -> 1
        architecture_part = random.choice([('num_encoder_layers', 'encoder_layers'),
                                           ('num_decoder_layers', 'decoder_layers')])

        self_encoded = self.encode()
        other_encoded = other_representation.encode()

        other_encoded[architecture_part[0]] = self_encoded[architecture_part[0]]
        other_encoded[architecture_part[1]] = self_encoded[architecture_part[1]]

        individual = LSTMAutoencoder(input_shape=self.input_shape,
                                     output_shape=self.output_shape,
                                     train_test_data=self.train_test_data,
                                     mutation_methods=self.mutation_methods,
                                     mutation_parameters=self.mutation_parameters)
        individual.decode(other_encoded)

        return individual

    def create_initial_population(self, population_size):
        """
        Create initial population for evolutionary algorithm.
        New individuals are created using random 5 mutations.

        :param population_size: Size of population.
        :return: Initialized population.
        """
        initial_population = []
        for i in range(population_size):
            individual = LSTMAutoencoder(input_shape=self.input_shape,
                                         output_shape=self.output_shape,
                                         train_test_data=self.train_test_data,
                                         mutation_methods=self.mutation_methods,
                                         mutation_parameters=self.mutation_parameters)
            individual._create_initial_representation()
            individual.mutate()
            initial_population.append(individual)

        return initial_population

    def evaluate(self):
        """
        Evaluate LSTM Autoencoder.
        If the model is not functional, the mutation will be used until the model is functional.
        mape - mean absolute percentage error
        num_params - number of model parameters
        memory_usage - megabytes used for building, training and evaluating of model

        :return: [mape, num_params, memory_usage_value]
        """
        x_test = self.train_test_data['x_test']
        y_test = self.train_test_data['y_test']

        tracemalloc.start()
        self._build_model()
        self._train_model()
        _, mse, mape = self.model.evaluate(x_test, y_test, verbose=1)
        current, peak = tracemalloc.get_traced_memory()
        memory_usage = peak / (1024 ** 2)

        num_params = self.model.count_params()

        result = [self.get_id(), [mape, num_params, memory_usage]]

        # If ant architecture is unable to work, returns nan, mutate it utils it starts work.
        if str(mape) == 'nan':
            result = [self.get_id(), [math.inf, math.inf, math.inf]]
            # self.mutate()
            # result = self.evaluate()

        return result

    def get_id(self):
        """
        Name getter.

        :return: __name__
        """
        return self.__name__
