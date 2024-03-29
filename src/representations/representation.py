from abc import ABC, abstractmethod


class Representation(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def encode(self):
        pass

    @abstractmethod
    def decode(self, encoded_representation):
        pass

    @abstractmethod
    def create_initial_population(self, population_size):
        pass

    @abstractmethod
    def evaluate(self):
        pass

    @abstractmethod
    def mutate(self):
        pass

    @abstractmethod
    def crossover(self, other_representation):
        pass

    @abstractmethod
    def display_representation(self):
        pass

    @abstractmethod
    def get_id(self):
        pass
