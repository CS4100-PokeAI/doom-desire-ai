from typing import Dict

from poke_env.environment.abstract_battle import AbstractBattle

class RewardCalculator():

    SPEED_TIER_COEFICIENT = 0.1
    HP_FRACTION_COEFICIENT = 0.4

    def __init__(
            self,
            *,
            victory_value: float = 1000.0,
            starting_value: float = 0.0,
            fainted_value: float = 10.0,
            hp_value: float = 5.0,
            number_of_pokemons: int = 6,
            status_value: float = 1.0,
            matchup_advantage_value: float = 3.0
    ):
        self._fainted_value = fainted_value
        self._hp_value = hp_value
        self._number_of_pokemons = number_of_pokemons
        self._starting_value = starting_value
        self._status_value = status_value
        self._victory_value = victory_value
        self._matchup_advantage_value = matchup_advantage_value

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

        return score

    def calc_reward(self, battle: AbstractBattle) -> float:
        """A helper function to compute rewards.

        The reward is computed by computing the value of a game state, and by comparing
        it to the last state.

        State values are computed by weighting different factor. Fainted pokemons,
        their remaining HP, inflicted statuses and winning are taken into account.

        For instance, if the last time this function was called for battle A it had
        a state value of 8 and this call leads to a value of 9, the returned reward will
        be 9 - 8 = 1.

        Consider a single battle where each player has 6 pokemons. No opponent pokemon
        has fainted, but our team has one fainted pokemon. Three opposing pokemons are
        burned. We have one pokemon missing half of its HP, and our fainted pokemon has
        no HP left.

        The value of this state will be:

        - With fainted value: 1, status value: 0.5, hp value: 1:
            = - 1 (fainted) + 3 * 0.5 (status) - 1.5 (our hp) = -1
        - With fainted value: 3, status value: 0, hp value: 1:
            = - 3 + 3 * 0 - 1.5 = -4.5

        :param battle: The battle for which to compute rewards.
        :type battle: AbstractBattle
        :param fainted_value: The reward weight for fainted pokemons. Defaults to 0.
        :type fainted_value: float
        :param hp_value: The reward weight for hp per pokemon. Defaults to 0.
        :type hp_value: float
        :param number_of_pokemons: The number of pokemons per team. Defaults to 6.
        :type number_of_pokemons: int
        :param starting_value: The default reference value evaluation. Defaults to 0.
        :type starting_value: float
        :param status_value: The reward value per non-fainted status. Defaults to 0.
        :type status_value: float
        :param victory_value: The reward value for winning. Defaults to 1.
        :type victory_value: float
        :return: The reward.
        :rtype: float
        """
        current_value = 0

        for mon in battle.team.values():
            current_value += mon.current_hp_fraction * self._hp_value  # Fraction of full HP value
            if mon.fainted:
                current_value -= self._fainted_value  # Subtract if fainted
            elif mon.status is not None:
                current_value -= self._status_value  # Any status subtracts

        current_value += (self._number_of_pokemons - len(battle.team)) * self._hp_value

        for mon in battle.opponent_team.values():
            current_value -= mon.current_hp_fraction * self._hp_value
            if mon.fainted:
                current_value += self._fainted_value
            elif mon.status is not None:
                current_value += self._status_value

        current_value -= (self._number_of_pokemons - len(battle.opponent_team)) * self._hp_value

        current_value += (self._estimate_matchup(battle.active_pokemon, battle.opponent_active_pokemon)) * self._matchup_advantage_value

        if battle.won:
            current_value += self._victory_value
        elif battle.lost:
            current_value -= self._victory_value

        return current_value

    @property
    def starting_value(self):
        return self._starting_value
