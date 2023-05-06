import os
import datetime


def load_pareto_fronts(file):
    pass


class LSTMAutoencoderLogger:
    def __init__(self):
        self.log_file = f"{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}.txt"
        self.log_path = os.path.join('logs', self.log_file)

        self.mutations_numbers = {
            'LSTM1_ADD':        0,
            'LSTM1_REMOVE':     0,
            'LSTM2_ADD':        0,
            'LSTM2_REMOVE':     0,
            'DENSE_ACT':        0,
            'LSTM_UNITS':       0,
            'LSTM_ACT':         0,
            'LSTM_REC_ACT':     0,
            'LSTM_DROPOUT':     0,
            'LSTM_BATCHNORM':   0,
        }

    def update_mutation_logs(self, mutation_name):
        self.mutations_numbers[mutation_name] = int(self.mutations_numbers[mutation_name]) + 1

    def reset_mutation_logs(self):
        self.mutations_numbers['LSTM1_ADD'] = 0
        self.mutations_numbers['LSTM1_REMOVE'] = 0
        self.mutations_numbers['LSTM2_ADD'] = 0
        self.mutations_numbers['LSTM2_REMOVE'] = 0
        self.mutations_numbers['DENSE_ACT'] = 0
        self.mutations_numbers['LSTM_UNITS'] = 0
        self.mutations_numbers['LSTM_ACT'] = 0
        self.mutations_numbers['LSTM_REC_ACT'] = 0
        self.mutations_numbers['LSTM_DROPOUT'] = 0
        self.mutations_numbers['LSTM_BATCHNORM'] = 0

    def log_mutations(self):
        with open(self.log_path, "a") as f:
            f.write(f"MUTATIONS:")
            f.write(f"LSTM1_ADD={self.mutations_numbers['LSTM1_ADD']},")
            f.write(f"LSTM1_REMOVE={self.mutations_numbers['LSTM1_REMOVE']},")
            f.write(f"LSTM2_ADD={self.mutations_numbers['LSTM2_ADD']},")
            f.write(f"LSTM2_REMOVE={self.mutations_numbers['LSTM2_REMOVE']},")
            f.write(f"DENSE_ACT={self.mutations_numbers['DENSE_ACT']},")
            f.write(f"LSTM_UNITS={self.mutations_numbers['LSTM_UNITS']},")
            f.write(f"LSTM_ACT={self.mutations_numbers['LSTM_ACT']},")
            f.write(f"LSTM_REC_ACT={self.mutations_numbers['LSTM_REC_ACT']},")
            f.write(f"LSTM_DROPOUT={self.mutations_numbers['LSTM_DROPOUT']},")
            f.write(f"LSTM_BATCHNORM={self.mutations_numbers['LSTM_BATCHNORM']}\n")
        self.reset_mutation_logs()

    def log_generation(self, generation_number):
        with open(self.log_path, "a") as f:
            f.write(f"GENERATION {generation_number} PARETO FRONT\n")

    def log_evaluation(self, individual_id, mse, num_params, memory_usage):
        with open(self.log_path, "a") as f:
            f.write(f'ID={individual_id},MSE={mse},MODEL_PARAMS={num_params},MEMORY_USAGE={memory_usage}\n')

    def log(self, message):
        with open(self.log_path, "a") as f:
            f.write(f"{message}\n")
