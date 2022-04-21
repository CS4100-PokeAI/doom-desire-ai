import numpy as np
from gym import Space
from gym.spaces import Box

from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.player.env_player import Gen8EnvSinglePlayer
from poke_env.player.openai_api import ObservationType
from poke_env.player.random_player import RandomPlayer


class ExampleRLPlayer2(Gen8EnvSinglePlayer):

    def calc_reward(self, last_battle, current_battle) -> float:
        return self.reward_computing_helper(
            current_battle, fainted_value=2.0, hp_value=1.0, victory_value=30.0
        )

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


class ExampleRLPlayer(Gen8EnvSinglePlayer):
    def calc_reward(self, last_battle, current_battle) -> float:
        return self.reward_computing_helper(
            current_battle, fainted_value=2.0, hp_value=1.0, victory_value=30.0
        )

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


# class ExampleRLPlayer(Gen8EnvSinglePlayer):
#     def embed_battle(self, battle):
#         # -1 indicates that the move does not have a base power
#         # or is not available
#         moves_base_power = -np.ones(4)
#         moves_dmg_multiplier = np.ones(4)
#         for i, move in enumerate(battle.available_moves):
#             moves_base_power[i] = (
#                 move.base_power / 100
#             )  # Simple rescaling to facilitate learning
#             if move.type:
#                 moves_dmg_multiplier[i] = move.type.damage_multiplier(
#                     battle.opponent_active_pokemon.type_1,
#                     battle.opponent_active_pokemon.type_2,
#                 )
#
#         # We count how many pokemons have not fainted in each team
#         remaining_mon_team = (
#             len([mon for mon in battle.team.values() if mon.fainted]) / 6
#         )
#         remaining_mon_opponent = (
#             len([mon for mon in battle.opponent_team.values() if mon.fainted]) / 6
#         )
#
#         # Final vector with 10 components
#         return np.concatenate(
#             [
#                 moves_base_power,
#                 moves_dmg_multiplier,
#                 [remaining_mon_team, remaining_mon_opponent],
#             ]
#         )
#
#     def compute_reward(self, battle) -> float:
#         return self.reward_computing_helper(
#             battle, fainted_value=2, hp_value=1, victory_value=30
#         )
#
#
# class MaxDamagePlayer(RandomPlayer):
#     def choose_move(self, battle):
#         # If the player can attack, it will
#         if battle.available_moves:
#             # Finds the best move among available ones
#             best_move = max(battle.available_moves, key=lambda move: move.base_power)
#             return self.create_order(best_move)
#
#         # If no attack is available, a random switch will be made
#         else:
#             return self.choose_random_move(battle)
