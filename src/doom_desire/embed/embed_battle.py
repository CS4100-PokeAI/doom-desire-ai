
# Define different kinds of embeddings for battles that can be used by RL agents

# Abstract
from abc import ABC, abstractmethod

import numpy as np
from numpy import ndarray
from typing import Tuple

from poke_env.environment.battle import Battle


class BattleEmbedding(ABC):

    @abstractmethod
    def embed_battle(self, battle: Battle) -> ndarray:
        pass

    @abstractmethod
    def input_shape(self) -> Tuple[int, int]:
        pass



class ExampleRLEmbedding(BattleEmbedding):
    """
    Simple Embedding with 10 components:

    Player Pokemon move base power x4
    Player Pokemon move damage multiplier x4
    Player Pokemon remaining
    Opponent Pokemon remaining
    """

    def input_shape(self) -> Tuple[int, int]:
        return (1, 10)

    def embed_battle(self, battle: Battle) -> ndarray:
        # -1 indicates that the move does not have a base power
        # or is not available
        moves_base_power = -np.ones(4)
        moves_dmg_multiplier = np.ones(4)
        for i, move in enumerate(battle.available_moves):
            moves_base_power[i] = (
                move.base_power / 100
            )  # Simple rescaling to facilitate learning
            if move.type:
                moves_dmg_multiplier[i] = move.type.damage_multiplier(
                    battle.opponent_active_pokemon.type_1,
                    battle.opponent_active_pokemon.type_2,
                )

        # We count how many pokemons have not fainted in each team
        remaining_mon_team = (
            len([mon for mon in battle.team.values() if mon.fainted]) / 6
        )
        remaining_mon_opponent = (
            len([mon for mon in battle.opponent_team.values() if mon.fainted]) / 6
        )

        # Final vector with 10 components
        return np.concatenate(
            [
                moves_base_power,
                moves_dmg_multiplier,
                [remaining_mon_team, remaining_mon_opponent],
            ]
        )


class ExpandedRLEmbedding(BattleEmbedding):

    def input_shape(self) -> Tuple[int, int]:
        return (1, 10)


    def embed_battle(self, battle: Battle) -> ndarray:

        player_team = []




        # -1 indicates that the move does not have a base power
        # or is not available
        moves_base_power = -np.ones(4)
        moves_dmg_multiplier = np.ones(4)
        for i, move in enumerate(battle.available_moves):
            moves_base_power[i] = (
                move.base_power / 100
            )  # Simple rescaling to facilitate learning
            if move.type:
                moves_dmg_multiplier[i] = move.type.damage_multiplier(
                    battle.opponent_active_pokemon.type_1,
                    battle.opponent_active_pokemon.type_2,
                )

        # We count how many pokemons have not fainted in each team
        remaining_mon_team = (
            len([mon for mon in battle.team.values() if mon.fainted]) / 6
        )
        remaining_mon_opponent = (
            len([mon for mon in battle.opponent_team.values() if mon.fainted]) / 6
        )

        # Final vector with 10 components
        return np.concatenate(
            [
                moves_base_power,
                moves_dmg_multiplier,
                [remaining_mon_team, remaining_mon_opponent],
            ]
        )
