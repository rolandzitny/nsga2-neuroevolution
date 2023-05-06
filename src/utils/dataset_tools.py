import numpy as np
import pandas as pd


def get_training_testing_windows(csv_file_pth, sequence_length, train_size):
    df = pd.read_csv(csv_file_pth)
    data = np.array(df)

    windows = data.reshape((100, 200, 6))

    windows = np.reshape(windows, (windows.shape[0], windows.shape[1], 6))
    x_windows = windows[:, :180, :]
    y_windows = windows[:, 180:, :]

    x_train = x_windows
    y_train = y_windows

    x_test = x_train[::2]
    y_test = y_train[::2]

    return x_train, y_train, x_test, y_test
