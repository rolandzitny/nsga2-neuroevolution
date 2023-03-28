import os
import datetime


class Logger:
    def __init__(self):
        self.log_file = f"{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}.txt"
        self.log_path = os.path.join(os.getcwd(), "logs", self.log_file)
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

    def log(self, message):
        with open(self.log_path, "a") as f:
            f.write(f"{message}\n")