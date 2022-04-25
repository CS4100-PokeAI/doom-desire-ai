from abc import ABC, abstractmethod
from typing import TypeVar, Tuple

from gym import Space

from poke_env.environment.abstract_battle import AbstractBattle

ObservationType = TypeVar("ObservationType")

class AbstractEmbedder(ABC):

    # @abstractmethod
    # def _embed_move(self, move):
    #     pass
    #
    # @abstractmethod
    # def _embed_mon(self, battle, mon):
    #     pass
    #
    # @abstractmethod
    # def _embed_opp_mon(self, battle, mon):
    #     pass

    @abstractmethod
    def embed_battle(self, battle: AbstractBattle) -> ObservationType:
        pass

    @abstractmethod
    def describe_embedding(self) -> Space:
        pass

    @abstractmethod
    def embedding_shape(self):
        pass

class AbstractFlatEmbedder(AbstractEmbedder):

    @abstractmethod
    def embedding_shape(self) -> Tuple[int, int]:  # TODO: This might not be right if there are nested embeddings
        pass

class AbstractFullEmbedder(AbstractEmbedder):

    @abstractmethod
    def describe_move_embedding(self) -> Space:
        pass

    @abstractmethod
    def describe_mon_embedding(self) -> Space:
        pass

    @abstractmethod
    def embed_battle(self, battle: AbstractBattle) -> ObservationType:
        pass


