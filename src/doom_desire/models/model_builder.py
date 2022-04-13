from abc import ABC, abstractmethod
from typing import List, Tuple

from tensorflow.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten


class ModelBuilder(ABC):

    def __init__(self) -> None:
        self.layers: List[any] = []

    @abstractmethod
    def build_model(self):
        pass

    @abstractmethod
    def add_layer(self) -> None:
        pass

class SequentialModelBuilder(ModelBuilder):

    def add_layer(self) -> None:
        pass

    def build_model(self) -> Sequential:

        return Sequential(self.layers)





class ExampleSequentialModelBuilder(SequentialModelBuilder):

    def build_shaped_model(self, input_shape: Tuple[int, int], output_size: int) -> Sequential:

        model = Sequential()
        model.add(Dense(128, activation="elu", input_shape=input_shape))

        # Our embedding have shape (1, 10), which affects our hidden layer
        # dimension and output dimension
        # Flattening resolve potential issues that would arise otherwise
        model.add(Flatten())
        model.add(Dense(64, activation="elu"))
        model.add(Dense(units=output_size, activation="linear"))

        return model
