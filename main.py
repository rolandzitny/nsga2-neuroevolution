from src.representations.lstm_autoencoder import LSTMAutoencoder


def main():
    print("Start ... ")
    arch1 = {'num_layers': 1, 'dense_activation': 'linear', 'layers': [{'type': 'LSTM', 'units': 1, 'activation': 'tanh', 'recurrent_activation': 'sigmoid', 'dropout': 0.2, 'use_batchnorm': True}]}
    arch2 = {'num_layers': 2, 'dense_activation': 'linear', 'layers': [{'type': 'LSTM', 'units': 2, 'activation': 'tanh', 'recurrent_activation': 'sigmoid', 'dropout': 0.2, 'use_batchnorm': True}, {'type': 'LSTM', 'units': 3, 'activation': 'tanh', 'recurrent_activation': 'sigmoid', 'dropout': 0.2, 'use_batchnorm': True}]}
    arch3 = {'num_layers': 3, 'dense_activation': 'linear', 'layers': [
        {'type': 'LSTM', 'units': 4, 'activation': 'tanh', 'recurrent_activation': 'sigmoid', 'dropout': 0.2,
         'use_batchnorm': True},
        {'type': 'LSTM', 'units': 5, 'activation': 'tanh', 'recurrent_activation': 'sigmoid', 'dropout': 0.2,
         'use_batchnorm': True}, {'type': 'LSTM', 'units': 6, 'activation': 'tanh', 'recurrent_activation': 'sigmoid', 'dropout': 0.2,
         'use_batchnorm': True}]}

    x = LSTMAutoencoder(input_shape=(180, 6), output_shape=6, max_lstm_layers=5)
    y = LSTMAutoencoder(input_shape=(180, 6), output_shape=6, max_lstm_layers=5)
    z = LSTMAutoencoder(input_shape=(180, 6), output_shape=6, max_lstm_layers=5)

    x.decode(arch1)
    y.decode(arch2)
    z.decode(arch3)

    model = z.build_model()
    model.summary()

if __name__ == '__main__':
    main()




