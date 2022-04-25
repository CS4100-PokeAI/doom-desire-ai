from typing import Tuple

import numpy as np
from gym import Space
from gym.spaces import Box

from doom_desire.embed.abstract_embedder import AbstractFlatEmbedder
from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.player.openai_api import ObservationType


class SimplePlusEmbedder(AbstractFlatEmbedder):
    """
    Simple Embedding with 10 components:

    Player Pokemon move base power x4
    Player Pokemon move damage multiplier x4
    Player Pokemon remaining
    Opponent Pokemon remaining
    """
    SPEED_TIER_COEFICIENT = 0.1
    HP_FRACTION_COEFICIENT = 0.4

    def _estimate_matchup(self, mon, opponent):
        score = max([opponent.damage_multiplier(t) for t in mon.types if t is not None])
        score -= max(
            [mon.damage_multiplier(t) for t in opponent.types if t is not None]
        )
        if mon.base_stats["spe"] > opponent.base_stats["spe"]:
            score += self.SPEED_TIER_COEFICIENT
        elif opponent.base_stats["spe"] > mon.base_stats["spe"]:
            score -= self.SPEED_TIER_COEFICIENT

        score += mon.current_hp_fraction * self.HP_FRACTION_COEFICIENT
        score -= opponent.current_hp_fraction * self.HP_FRACTION_COEFICIENT

        # Bound the outputs to +-2
        return 2 if score > 2 else -2 if score < -2 else score

    def _stat_estimation(self, mon, stat):
        # Stats boosts value
        if mon.boosts[stat] > 1:
            boost = (2 + mon.boosts[stat]) / 2
        else:
            boost = 2 / (2 - mon.boosts[stat])
        estimate = ((2 * mon.base_stats[stat] + 31) + 5) * boost

        return 2 if estimate > 2 else estimate

    def embed_battle(self, battle: AbstractBattle) -> ObservationType:
        # -1 indicates that the move does not have a base power
        # or is not available

        active = battle.active_pokemon
        opponent = battle.opponent_active_pokemon

        matchup_estimates = []
        matchup_estimates.append(self._estimate_matchup(active, opponent))

        # For moves, embed base power, damage multiplier, and physicial-special ratio for current mons

        physical_special_ratio = [
            self._stat_estimation(active, "atk") / self._stat_estimation(opponent, "def"),  # physical_ratio
            self._stat_estimation(active, "spa") / self._stat_estimation(opponent, "spd")]  # special_ratio

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

        final_vector = np.concatenate(
            [
                matchup_estimates,
                physical_special_ratio,
                moves_base_power,
                moves_dmg_multiplier,
                [fainted_mon_team, fainted_mon_opponent],
            ]
        )
        return np.float32(final_vector)

    def describe_embedding(self) -> Box:
        """
        Things embedded for a 46 total
            4x moves (30 each, 120 total)
            is active, current hp, fainted, is dynamaxed (1 each, 4 total)
            base stat values (6 total) (actual not known)
            status (7 total)
            types (36 total)
        """
        low_high_dict = {'matchup_estimate': {'low': [-2], 'high': [2], 'times': 1},
                         'ph_sp_ratio': {'low': [0], 'high': [2], 'times': 2},
                         'move_power': {'low': [-1], 'high': [3], 'times': 4},
                         'dmg_multi': {'low': [0], 'high': [4], 'times': 4},
                         'fained_mons': {'low': [0], 'high': [1], 'times': 2},
                         }

        low = [sub_dict['low'] for sub_dict in low_high_dict.values() for _ in range(sub_dict['times'])]
        low = [item for sublist in low for item in sublist]

        high = [sub_dict['high'] for sub_dict in low_high_dict.values() for _ in range(sub_dict['times'])]
        high = [item for sublist in high for item in sublist]

        return Box(
            np.array(low, dtype=np.float32),
            np.array(high, dtype=np.float32),
            dtype=np.float32,
        )

    def embedding_shape(self)-> Tuple[int, int]:
        return (1, 13)
