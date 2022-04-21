from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
from gym import Space
from gym.spaces import Box

from doom_desire.helpers.abstract_embedder import AbstractEmbedder
from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.player.openai_api import ObservationType


class SimpleEmbedder(AbstractEmbedder):
    """
    Simple Embedding with 10 components:

    Player Pokemon move base power x4
    Player Pokemon move damage multiplier x4
    Player Pokemon remaining
    Opponent Pokemon remaining
    """

    def embed_battle(self, battle: AbstractBattle) -> ObservationType:
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

        # We count how many pokemons have fainted in each team
        fainted_mon_team = len([mon for mon in battle.team.values() if mon.fainted]) / 6
        fainted_mon_opponent = (
            len([mon for mon in battle.opponent_team.values() if mon.fainted]) / 6
        )

        # Final vector with 10 components
        final_vector = np.concatenate(
            [
                moves_base_power,
                moves_dmg_multiplier,
                [fainted_mon_team, fainted_mon_opponent],
            ]
        )
        return np.float32(final_vector)


    def describe_embedding(self) -> Space:
        low = [-1, -1, -1, -1, 0, 0, 0, 0, 0, 0]
        high = [3, 3, 3, 3, 4, 4, 4, 4, 1, 1]
        return Box(
            np.array(low, dtype=np.float32),
            np.array(high, dtype=np.float32),
            dtype=np.float32,
        )


    def embedding_shape(self)-> Tuple[int, int]:
        return (1, 10)
