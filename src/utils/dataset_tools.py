import numpy as np
import pandas as pd


def get_training_testing_windows(csv_file_pth, sequence_length, train_size, size_train=None, size_test=None):
    df = pd.read_csv(csv_file_pth)
    data = np.array(df)
    train_size = int(len(data) * train_size)

    x_train = []
    x_test = []

    for i in range(sequence_length, train_size):
        x_train.append(data[i-sequence_length:i, :])

    for i in range(train_size, len(data) - sequence_length):
        x_test.append(data[i-sequence_length:i, :])

    x_train = np.array(x_train)
    x_test = np.array(x_test)

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 6))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 6))

    if size_train is not None:
        x_train = x_train[:size_train]

    if size_test is not None:
        x_test = x_test[:size_test]

    x_train_processed = x_train[:, :180, :]
    y_train_processed = x_train[:, 180:, :]

    x_test_processed = x_test[:, :180, :]
    y_test_processed = x_test[:, 180:, :]

    train_test_data = {
        'x_train': x_train_processed,
        'y_train': y_train_processed,
        'x_test': x_test_processed,
        'y_test': y_test_processed
    }

    return x_train_processed, y_train_processed, x_test_processed, y_test_processed