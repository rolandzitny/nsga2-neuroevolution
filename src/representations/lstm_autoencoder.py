import random
import tensorflow as tf
from keras.optimizers import RMSprop
from keras.losses import MeanSquaredError
from keras.layers import LSTM, Dropout, BatchNormalization, TimeDistributed, Dense


class LSTMAutoencoder:
    def __init__(self, input_shape, output_shape, max_lstm_layers):
        """
        Create dictionary representation class for LSTM AutoEncoder used for multivariate signal forecasting.
        Whole idea is to forecast 20 new timestamps/rows from initial 180 timestamps/rows of multivariate signal,
        so we need to use return sequence parameter of LSTM layer equal True.

        Example of final representation:

        {
            "num_layers": 3,
            "dense_activation": "sigmoid",
            "layers": [
                {
                    "type": "LSTM",
                    "units": 32,
                    "activation": "tanh",
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
                },
                {
                    "type": "LSTM",
                    "units": 128,
                    "activation": "tanh",
                    "recurrent_activation": "sigmoid",
                    "dropout": false,
                    "use_batchnorm": false
                }
            ]
        }

         If dropout is false, there is no Dropout layer. If dropout is a number in the range 0 to 1, a Dropout layer
         is added with the specified dropout rate. The use_batchnorm parameter is used to indicate whether a
         BatchNormalization layer should be added after LSTM layer.

        :param input_shape: input shape e.g. (180, 6), 180 rows with 6 values
        :param output_shape: number of forecasted values, it is number of units in last dense layer
        :param max_lstm_layers: defines maximal number of lstm layers in architecture, used in mutation
        """
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.max_lstm_layers = max_lstm_layers

        self.num_layers = None
        self.dense_activation = None
        self.layers = []
        self.model = None

    def create_base(self):
        """
        Creates basic LSTM Neural Network with just one LSTM layer.
        In the model we want to forecast e.g. from 180 timestamps/rows next 20, so we need to use
        return sequence parameter of LSTM layer equal True.
        """
        self.num_layers = 1
        self.dense_activation = 'linear'
        self.layers = [
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

        :return: architecture dictionary
        """
        encoded_architecture = {
            "num_layers": self.num_layers,
            "dense_activation": self.dense_activation,
            "layers": self.layers
        }
        return encoded_architecture

    def decode(self, encoded_architecture):
        """
        Load encoded architecture into inner representation in class (update class variables).

        :param encoded_architecture:
        """
        self.num_layers = int(encoded_architecture["num_layers"])
        self.dense_activation = encoded_architecture["dense_activation"]
        self.layers = encoded_architecture["layers"]

    def display_architecture(self):
        """
        Display representation of architecture.
        """
        print("")
        print("Input shape: ", self.input_shape)
        print("Output shape: ", self.output_shape)
        print("Number of LSTM layers:", self.num_layers)
        print("Activation function of Dense layer:", self.dense_activation)
        print("Architecture:")
        for layer in self.layers:
            print(layer)
        print("")

    def build_model(self):
        """
        Create Tensorflow model from representation.
        When we are forecasting multivariate signal multiple steps into future
        e.g. from 180 samples forecast 20 steps in future last layer is TimeDistributed(Dense()) layer.
        As optimizer is used RMSprop commonly used in recurrent nn and as loss function Mean Squared Error.
        Adam and MAE is also possible to use.

        :return: compiled keras model
        """
        inputs = tf.keras.Input(shape=self.input_shape)

        architecture = inputs

        for i, layer_params in enumerate(self.layers):
            # Create whole architecture with LSTM, Dropout and BatchNormalization layers
            if layer_params["type"] == "LSTM":
                # LSTM layer
                # return_sequences=True because we need to forecast multiple values
                architecture = LSTM(layer_params["units"], activation=layer_params["activation"],
                                    recurrent_activation=layer_params["recurrent_activation"],
                                    return_sequences=True)(architecture)

                # Dropout layer
                if layer_params["dropout"]:
                    architecture = Dropout(layer_params["dropout"])(architecture)

                # BatchNormalization layer
                if layer_params["use_batchnorm"]:
                    architecture = BatchNormalization()(architecture)

            else:
                raise ValueError("Unsupported layer type")

            # Add last layer (Dense), which is TimeDistributed
            # it is used when forcasting multiple values into future
            if i == self.num_layers - 1:
                architecture = TimeDistributed(Dense(self.output_shape, activation=self.dense_activation))(architecture)

        # Create Tensorflow model
        self.model = tf.keras.Model(inputs=inputs, outputs=architecture)
        self.model.compile(optimizer=RMSprop(learning_rate=0.001),
                           loss=MeanSquaredError(),
                           metrics=['mse', 'mae'])
        return self.model

    def _add_lstm_mutation(self, hyper_random):
        """
        Adds one more LSTM layer on random space of architecture if maximal number of layers is not reached.

        :param hyper_random: True/False -> whether to randomly set hyper-parameters of new LST layer
        or copy hyper-parameters of last LSTM layer.
        :return: mutated LSTMAutoencoder object or False if number of maximal number of layers has been reached
        """
        if self.num_layers < self.max_lstm_layers:
            encoded_architecture = self.encode()
            encoded_architecture['num_layers'] = str(int(encoded_architecture['num_layers']) + 1)

            # Randomly choose hyper-parameters of new LSTM layer
            if hyper_random:
                lstm_layers = encoded_architecture['layers']

                units = [8, 16, 32, 64, 128, 256]
                activation = ['sigmoid', 'tanh', 'relu', 'softmax', 'linear']
                rec_activation = ['sigmoid', 'hard_sigmoid', 'tanh', 'relu', 'linear']
                dropout_choices = [False, 0.2, 0.3, 0.4, 0.5]
                batch_norm_choices = [True, False]

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
                encoded_architecture['layers'] = lstm_layers
                mutated_lstm_autoencoder = LSTMAutoencoder(self.input_shape, self.output_shape, self.max_lstm_layers)
                mutated_lstm_autoencoder.decode(encoded_architecture)
                return mutated_lstm_autoencoder

            # Copy last lstm layer and add it into architecture
            else:
                lstm_layers = encoded_architecture['layers']
                # Get last LSTM layer
                new_lstm_layer = lstm_layers[-1]
                random_index = random.randint(0, len(lstm_layers))
                lstm_layers.insert(random_index, new_lstm_layer)
                encoded_architecture['layers'] = lstm_layers
                mutated_lstm_autoencoder = LSTMAutoencoder(self.input_shape, self.output_shape, self.max_lstm_layers)
                mutated_lstm_autoencoder.decode(encoded_architecture)
                return mutated_lstm_autoencoder
        else:
            # If mutation is not possible (max lstm layers) return False and do another mutation
            return False

    def _remove_lstm_mutation(self):
        """
        Remove random LSTM layer from architecture.

        :return: mutated LSTMAutoencoder object or False if number of maximal number of layers has been reached
        """
        if self.num_layers > 1:
            encoded_architecture = self.encode()
            encoded_architecture['num_layers'] = str(int(encoded_architecture['num_layers']) - 1)
            lstm_layers = encoded_architecture['layers']
            # Remove random LSTM layer from architecture
            index = random.randint(0, len(lstm_layers) - 1)
            lstm_layers.pop(index)
            encoded_architecture['layers'] = lstm_layers
            mutated_lstm_autoencoder = LSTMAutoencoder(self.input_shape, self.output_shape, self.max_lstm_layers)
            mutated_lstm_autoencoder.decode(encoded_architecture)
            return mutated_lstm_autoencoder
        else:
            return False

    def _dense_layer_mutation(self):
        """
        Mutate activation function of last Dense layer.

        :return: mutated LSTMAutoencoder object
        """
        encoded_architecture = self.encode()
        dense_activation = encoded_architecture['dense_activation']
        dense_activation_functions = ['sigmoid', 'tanh', 'softmax', 'linear']
        # Remove used activation function to avoid repeating/neutral mutation
        dense_activation_functions.remove(dense_activation)
        encoded_architecture['dense_activation'] = random.choice(dense_activation_functions)
        mutated_lstm_autoencoder = LSTMAutoencoder(self.input_shape, self.output_shape, self.max_lstm_layers)
        mutated_lstm_autoencoder.decode(encoded_architecture)
        return mutated_lstm_autoencoder

    def _lstm_units_mutation(self, change_rate):
        """
        Mutate number of LSTM units in random LSTM layer.

        :param change_rate: rate of units change, e.g. 0.2 mean range -20%/+20% of previous value
        :return: mutated LSTMAutoencoder object
        """
        encoded_architecture = self.encode()
        lstm_layers = encoded_architecture['layers']
        random_lstm_layer_idx = random.randint(0, len(lstm_layers)-1)
        random_lstm_layer = lstm_layers[random_lstm_layer_idx]
        current_lstm_units = random_lstm_layer['units']
        lower_bound = int(current_lstm_units * (1 - change_rate))
        upper_bound = int(current_lstm_units * (1 + change_rate))
        new_lstm_units = random.randint(lower_bound, upper_bound)
        lstm_layers[random_lstm_layer_idx]['units'] = new_lstm_units
        encoded_architecture['layers'] = lstm_layers
        mutated_lstm_autoencoder = LSTMAutoencoder(self.input_shape, self.output_shape, self.max_lstm_layers)
        mutated_lstm_autoencoder.decode(encoded_architecture)
        return mutated_lstm_autoencoder

    def _lstm_activation_mutation(self):
        """
        Mutate activation function in random LSTM layer.

        :return: mutated LSTMAutoencoder object
        """
        encoded_architecture = self.encode()
        lstm_layers = encoded_architecture['layers']
        random_lstm_layer_idx = random.randint(0, len(lstm_layers) - 1)
        random_lstm_layer = lstm_layers[random_lstm_layer_idx]
        current_lstm_activation = random_lstm_layer['activation']
        activation = ['sigmoid', 'tanh', 'relu', 'softmax', 'linear']
        activation.remove(current_lstm_activation)
        lstm_layers[random_lstm_layer_idx]['activation'] = random.choice(activation)
        encoded_architecture['layers'] = lstm_layers
        mutated_lstm_autoencoder = LSTMAutoencoder(self.input_shape, self.output_shape, self.max_lstm_layers)
        mutated_lstm_autoencoder.decode(encoded_architecture)
        return mutated_lstm_autoencoder

    def _lstm_rec_activation_mutation(self):
        """
        Mutate recurrent activation function in random LSTM layer.

        :return: mutated LSTMAutoencoder object
        """
        encoded_architecture = self.encode()
        lstm_layers = encoded_architecture['layers']
        random_lstm_layer_idx = random.randint(0, len(lstm_layers) - 1)
        random_lstm_layer = lstm_layers[random_lstm_layer_idx]
        current_lstm_rec_activation = random_lstm_layer['recurrent_activation']
        rec_activation = ['sigmoid', 'hard_sigmoid', 'tanh', 'relu', 'linear']
        rec_activation.remove(current_lstm_rec_activation)
        lstm_layers[random_lstm_layer_idx]['recurrent_activation'] = random.choice(rec_activation)
        encoded_architecture['layers'] = lstm_layers
        mutated_lstm_autoencoder = LSTMAutoencoder(self.input_shape, self.output_shape, self.max_lstm_layers)
        mutated_lstm_autoencoder.decode(encoded_architecture)
        return mutated_lstm_autoencoder

    def _lstm_dropout_mutation(self):
        """
        Mutate dropout parameter in random LSTM layer.

        :return: mutated LSTMAutoencoder object
        """
        encoded_architecture = self.encode()
        lstm_layers = encoded_architecture['layers']
        random_lstm_layer_idx = random.randint(0, len(lstm_layers) - 1)
        random_lstm_layer = lstm_layers[random_lstm_layer_idx]
        current_lstm_dropout = random_lstm_layer['dropout']
        dropouts = [False, 0.1, 0.2, 0.3, 0.4, 0.5]
        dropouts.remove(current_lstm_dropout)
        lstm_layers[random_lstm_layer_idx]['dropout'] = random.choice(dropouts)
        encoded_architecture['layers'] = lstm_layers
        mutated_lstm_autoencoder = LSTMAutoencoder(self.input_shape, self.output_shape, self.max_lstm_layers)
        mutated_lstm_autoencoder.decode(encoded_architecture)
        return mutated_lstm_autoencoder

    def _lstm_batch_norm_mutation(self):
        """
        Mutate use_batchnorm parameter in random LSTM layer.

        :return: mutated LSTMAutoencoder object
        """
        encoded_architecture = self.encode()
        lstm_layers = encoded_architecture['layers']
        random_lstm_layer_idx = random.randint(0, len(lstm_layers) - 1)
        random_lstm_layer = lstm_layers[random_lstm_layer_idx]
        current_lstm_use_batch = random_lstm_layer['use_batchnorm']
        batch_norm_uses = [True, False]
        batch_norm_uses.remove(current_lstm_use_batch)
        lstm_layers[random_lstm_layer_idx]['use_batchnorm'] = random.choice(batch_norm_uses)
        encoded_architecture['layers'] = lstm_layers
        mutated_lstm_autoencoder = LSTMAutoencoder(self.input_shape, self.output_shape, self.max_lstm_layers)
        mutated_lstm_autoencoder.decode(encoded_architecture)
        return mutated_lstm_autoencoder

    def mutate(self, mutation_number, unique, hyper_random, change_rate):
        """
        Mutate current LSTM Autoencoder and returns new mutated object.
        This method always do some mutation, it means in case of failed mutation method it will randomly choose
        from method which will not fail.

        # TODO choose/remove which method to use or not -> parameter[list] which will update mutation_methods list.

        :param mutation_number: How many times to mutate one architecture, do not use more than 8
        :param unique: True/False -> whether to choose unique mutations or not when we choose more than 1 mutation
        :param hyper_random: True/False -> whether to set hyper-parameters of new LSTM layer created using
        _add_lstm_mutation at random or copy last layer
        :param change_rate: change rate for LSTM unit number
        :return: LSTMAutoencoder object with new mutated architecture
        """
        # Mutation methods in form: (method, parameters dictionary)
        mutation_methods = [(self._add_lstm_mutation, {'hyper_random': hyper_random}),
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

    def crossover(self, other_lstm_autoencoder):
        self_encoded = self.encode()
        other_encoded = other_lstm_autoencoder.encode()

        num_layers = min(self.max_lstm_layers, self.num_layers, other_encoded['num_layers'])
        crossover_point = random.randint(0, num_layers - 1)

        # Swap architectures at crossover point
        self_layers = self_encoded['layers'][:crossover_point] + other_encoded['layers'][crossover_point:]
        other_layers = other_encoded['layers'][:crossover_point] + self_encoded['layers'][crossover_point:]

        crossed1_architecture = self_encoded
        crossed1_architecture['num_layers'] = len(self_layers)
        crossed1_architecture['layers'] = self_layers

        crossed2_architecture = other_encoded
        crossed2_architecture['num_layers'] = len(other_layers)
        crossed2_architecture['layers'] = other_layers

        crossed1_lstm_ae = LSTMAutoencoder(self.input_shape, self.output_shape, self.max_lstm_layers)
        crossed1_lstm_ae.decode(crossed1_architecture)

        crossed2_lstm_ae = LSTMAutoencoder(self.input_shape, self.output_shape, self.max_lstm_layers)
        crossed2_lstm_ae.decode(crossed2_architecture)

        return crossed1_lstm_ae, crossed2_lstm_ae
