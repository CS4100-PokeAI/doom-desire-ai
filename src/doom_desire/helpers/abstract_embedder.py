from abc import ABC, abstractmethod

from gym import Space

from poke_env.environment.abstract_battle import AbstractBattle


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
    def embed_battle(self, battle: AbstractBattle):
        pass

    @abstractmethod
    def describe_embedding(self) -> Space:
        pass

    @abstractmethod
    def embedding_shape(self):
        pass
