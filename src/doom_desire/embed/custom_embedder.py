from typing import Tuple

import numpy as np
from gym import Space
from gym.spaces import Box

from doom_desire.embed.abstract_embedder import AbstractEmbedder
from poke_env.data import GEN_TO_MOVES, GEN_TO_POKEDEX
from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.environment.field import Field
from poke_env.environment.move_category import MoveCategory
from poke_env.environment.pokemon_type import PokemonType
from poke_env.environment.side_condition import SideCondition
from poke_env.environment.status import Status
from poke_env.environment.target_type import TargetType
from poke_env.environment.volatile_status import VolatileStatus
from poke_env.environment.weather import Weather
from poke_env.player.openai_api import ObservationType


class CustomEmbedder(AbstractEmbedder):

    def __init__(self, gen=8, priority=0):

        # TODO: implement by creating priority tiers with which we should embed different aspects of the game
        self.priority = priority
        self.gen = gen

        # Store all possible game-related knowledge, so that we can can embed battle states. The tuples are key where
        # we retrieve the classes, the class, and whether poke_env supports returning the class (as opposed to string)
        self._knowledge = {}
        sets = [
            ('Field', Field, False),
            ('SideCondition', SideCondition, False),
            ('Status', Status, True),
            ('Weather', Weather, True),
            ('PokemonType', PokemonType, True),
            ('MoveCategory', MoveCategory, True),
            ('TargetType', TargetType, False),
            ('VolatileStatus', VolatileStatus, False),
        ]

        for key, klass, supported in sets:
            if supported:
                self._knowledge[key] = list(klass._member_map_.values())
            else:
                self._knowledge[key] = list(
                    map(lambda x: x.name.lower().replace("_", ""), list(klass._member_map_.values())))

        self._knowledge['Move'] = list(GEN_TO_MOVES[gen].keys())
        self._knowledge['Pokemon'] = list(GEN_TO_POKEDEX[gen].keys())
        self._knowledge['Ability'] = list(set([
            ability for sublist in map(lambda x: x['abilities'].values(), GEN_TO_POKEDEX[gen].values()) for ability in
            sublist
        ]))

        # These are the lengths of the embeddings of each function.
        self.MOVE_LEN = 30  # for gen8
        self.MON_LEN = 172
        self.OPP_MON_LEN = 173
        self.BATTLE_LEN = 2113

    # Returns an array of an embedded move; could be precomputed (total length of 30 with all shortenings)
    def _embed_move(self, move):
        """
        Things embedded for a 30 total
            accuracy, base power, priority (1 each, 3 total)
            move category OHE (3 total)
            move type OHE (18 total)
            stat boosts (5 total)
            chance of secondary (1 total)
        """

        # If the move is None or empty, return a negative array (filled w/ -1's)
        if move is None or move.is_empty: return [-1] * self.MOVE_LEN

        embeddings = []

        # Encode other properties
        embeddings.append([  # Total length of 3
            move.accuracy,
            move.base_power,
            # int(move.breaks_protect),
            # move.crit_ratio,
            # move.current_pp,
            # move.damage,
            # move.drain,
            # move.expected_hits,
            # int(move.force_switch),
            # move.heal,
            # int(move.ignore_ability),
            # int(move.ignore_defensive),
            # int(move.ignore_evasion),
            # 1 if move.ignore_immunity else 0,
            # move.n_hit[0] if move.n_hit else 1, # minimum times the move hits
            # move.n_hit[1] if move.n_hit else 1, # maximum times the move hits
            move.priority,
            # move.recoil,
            # int(move.self_destruct is not None),
            # int(move.self_switch is not None),
            # int(move.steals_boosts),
            # int(move.thaws_target),
            # int(move.use_target_offensive),
        ])

        # OHE Move, Category, Defensive Category, Move Type
        # embeddings.append([1 if move.id == m else 0 for m in self._knowledge['Move']])
        # Length of 3
        embeddings.append([1 if move.category == category else 0 for category in self._knowledge['MoveCategory']])
        # embeddings.append([1 if move.defensive_category == category else 0 for category in self._knowledge['MoveCategory']])
        # Length of 18
        embeddings.append([1 if move.type == pokemon_type else 0 for pokemon_type in self._knowledge['PokemonType']])

        # OHE Fields, SC, Weather (bad coding -- assumes field name will be move name, and uses string manipulation)
        # embeddings.append([1 if move.id == field else 0 for field in self._knowledge['Field']])
        # embeddings.append([1 if move.side_condition == sc else 0 for sc in self._knowledge['SideCondition']])
        # embeddings.append([1 if move.weather == weather else 0 for weather in self._knowledge['Weather']])

        # OHE Targeting Types
        # embeddings.append([1 if move.deduced_target and move.deduced_target.lower() == tt else 0 for tt in self._knowledge['TargetType']])

        # OHE Volatility Statuses
        # volatility_status_embeddings = []
        # for vs in self._knowledge['VolatileStatus']:
        #     if vs == move.volatile_status: volatility_status_embeddings.append(1)
        #     elif move.secondary and vs in list(map(lambda x: x.get('volatilityStatus', '').lower(), move.secondary)): volatility_status_embeddings.append(1)
        #     else: volatility_status_embeddings.append(0)
        # embeddings.append(volatility_status_embeddings)

        # OHE Statuses
        # status_embeddings = []
        # for status in self._knowledge['Status']:
        #     if status == move.status: status_embeddings.append(1)
        #     elif move.secondary and status in list(map(lambda x: x.get('status', ''), move.secondary)): status_embeddings.append(1)
        #     else: status_embeddings.append(0)
        # embeddings.append(status_embeddings)

        # OHE Boosts to the move's target (which sometimes are self-boosts)
        # boost_embeddings = {'atk': 0, 'def': 0, 'spa': 0, 'spd': 0, 'spe': 0, 'evasion': 0, 'accuracy': 0}
        # if move.boosts:
        #     for stat in move.boosts: boost_embeddings[stat] += move.boosts[stat]
        # elif move.secondary:
        #     for x in move.secondary:
        #         for stat in x.get('boosts', {}): boost_embeddings[stat] += x['boosts'][stat]
        # embeddings.append(boost_embeddings.values())

        # Add Self-Boosts, total length of 7
        self_boost_embeddings = {'atk': 0, 'def': 0, 'spa': 0, 'spd': 0, 'spe': 0, 'evasion': 0, 'accuracy': 0}
        if move.self_boost:
            for stat in move.self_boost:
                self_boost_embeddings[stat] += move.self_boost[stat]
        elif move.secondary:
            for x in move.secondary:
                for stat in x.get('self', {}).get('boosts', {}): self_boost_embeddings[stat] += x['self']['boosts'][
                    stat]
        embeddings.append(list(self_boost_embeddings.values())[:-2])  # ignore evasion and accuracy boosts

        # Introduce the chance of a secondary effect happening, total length of only 1
        chance = 0
        for x in move.secondary:
            chance = max(chance, x.get('chance', 0))
        embeddings.append([chance])

        # Flatten the arrays
        return [item for sublist in embeddings for item in sublist]

    def _describe_move_embedding(self):
        """
        Things embedded for a 30 total
            accuracy, base power, priority (1 each, 3 total)
            move category OHE (3 total)
            move type OHE (18 total)
            stat boosts (5 total)
            chance of secondary (1 total)
        """
        low_high_dict = {'accuracy': {'low': -1, 'high': 1, 'times': 1},
                         'base_power': {'low': -1, 'high': 300, 'times': 1},
                         'priority': {'low': -7, 'high': 6, 'times': 1},
                         'move_category': {'low': -1, 'high': 1, 'times': 3},
                         'type': {'low': -1, 'high': 1, 'times': 18},
                         'stat_boost': {'low': -6, 'high': 6, 'times': 5},
                         'secondary_chance': {'low': -1, 'high': 1, 'times': 1}
                         }

        low_move  =  [sub_dict['low']  for sub_dict in low_high_dict.values() for _ in range(sub_dict['times'])]
        high_move  = [sub_dict['high'] for sub_dict in low_high_dict.values() for _ in range(sub_dict['times'])]
        return low_move, high_move

    # Returns an array of an embedded mon; could be precomputed per battle
    def _embed_mon(self, battle, mon):
        """
        Things embedded for a 172 total
            4x moves (30 each, 120 total)
            is active, current hp, fainted, is dynamaxed (1 each, 4 total)
            stat values (5 total) (doesn't include HP)
            status (7 total)
            types (36 total)
        """
        embeddings = []

        # OHE mons, for dex number
        # embeddings.append([1 if mon == pokemon else 0 for pokemon in self._knowledge['Pokemon']])

        # OHE abilities, for all possible abilities
        # embeddings.append([1 if mon.ability == ability else 0 for ability in self._knowledge['Ability']])

        # TODO: OHE items

        # Append moves to embedding (and account for the fact that the mon might have <4 moves)
        for move in (list(mon.moves.values()) + [None, None, None, None])[:4]:
            embeddings.append(self._embed_move(move))

        # Add whether the mon is active, the current hp, whether its fainted, its level, its weight and whether its recharging or preparing
        embeddings.append([
            int(mon.active),
            mon.current_hp,
            int(mon.fainted),
            # mon.level,  # they will all have the same level
            # mon.weight,
            # int(mon.must_recharge),
            # 1 if mon.preparing else 0,
            int(mon.is_dynamaxed),
        ])

        # Add stats and boosts
        embeddings.append(mon.stats.values())
        # embeddings.append(mon.boosts.values())  # only the current can have boosts anyway so don't do for all

        # Add status (one-hot encoded)
        embeddings.append([1 if mon.status == status else 0 for status in self._knowledge['Status']])

        # Add Types (one-hot encoded)
        embeddings.append([1 if mon.type_1 == pokemon_type else 0 for pokemon_type in self._knowledge['PokemonType']])
        embeddings.append([1 if mon.type_2 == pokemon_type else 0 for pokemon_type in self._knowledge['PokemonType']])

        # Add whether the mon is trapped or forced to switch. But first, find the index
        # index = None
        # if mon in battle.active_pokemon: index = 0 if battle.active_pokemon[0] == mon else 1
        # embeddings.append([
        #     1 if index and battle.trapped[index] else 0,
        #     1 if index and battle.force_switch[index] else 0,
        # ])

        # Flatten all the lists into a Nx1 list
        return [item for sublist in embeddings for item in sublist]

    def _describe_mon_embedding(self):
        """
        Things embedded for a 172 total
            4x moves (30 each, 120 total)
            is active, current hp, fainted, is dynamaxed (1 each, 4 total)
            stat values (5 total) (doesn't include HP)
            status (7 total)
            types (36 total)
        """
        low_move, high_move = self._describe_move_embedding()

        low_high_dict = {'moves': {'low': low_move, 'high': high_move, 'times': 4},
                         'active': {'low': [0], 'high': [1], 'times': 1},
                         'current_hp': {'low': [0], 'high': [1000], 'times': 1},
                         'fainted': {'low': [0], 'high': [1], 'times': 1},
                         'dynamaxed': {'low': [0], 'high': [1], 'times': 1},
                         'stat_val': {'low': [0], 'high': [1000], 'times': 5},
                         'status': {'low': [0], 'high': [1], 'times': 7},
                         'type1': {'low': [0], 'high': [1], 'times': 18},
                         'type2': {'low': [0], 'high': [1], 'times': 18},
                         }

        low_mon  = [sub_dict['low']  for sub_dict in low_high_dict.values() for _ in range(sub_dict['times'])]
        low_mon = [item for sublist in low_mon for item in sublist]

        high_mon = [sub_dict['high'] for sub_dict in low_high_dict.values() for _ in range(sub_dict['times'])]
        high_mon = [item for sublist in high_mon for item in sublist]

        return low_mon, high_mon

    def _embed_opp_mon(self, battle, mon):
        """
        Things embedded for a 173 total
            4x moves (30 each, 120 total)
            is active, current hp, fainted, is dynamaxed (1 each, 4 total)
            base stat values (6 total) (actual not known)
            status (7 total)
            types (36 total)
        """
        embeddings = []

        # Add whether the mon is active, the current hp, whether its fainted, its level, its weight and whether its recharging or preparing
        embeddings.append([
            int(mon.active),  # This mon is on the field now
            # int(mon in battle.opponent_team.values()), # This mon was brought
            mon.current_hp,
            int(mon.fainted),
            # mon.level,
            # mon.weight,
            # int(mon.must_recharge),
            # 1 if mon.preparing else 0,
            int(mon.is_dynamaxed),
        ])

        # Append moves to embedding (and account for the fact that the mon might have <4 moves, or we don't know of them)
        for move in (list(mon.moves.values()) + [None, None, None, None])[:4]:
            embeddings.append(self._embed_move(move))

        # OHE mons
        # embeddings.append([1 if mon == pokemon else 0 for pokemon in self._knowledge['Pokemon']])

        # OHE possible abilities
        # embeddings.append([1 if ability in GEN_TO_POKEDEX[self.gen][mon.species].abilities else 0 for ability in self._knowledge['Ability']])

        # TODO: OHE items

        # Add stats and boosts
        embeddings.append(mon.base_stats.values())
        # embeddings.append(mon.boosts.values())

        # Add status (one-hot encoded)
        embeddings.append([1 if mon.status == status else 0 for status in self._knowledge['Status']])

        # Add Types (one-hot encoded)
        embeddings.append([1 if mon.type_1 == pokemon_type else 0 for pokemon_type in self._knowledge['PokemonType']])
        embeddings.append([1 if mon.type_2 == pokemon_type else 0 for pokemon_type in self._knowledge['PokemonType']])

        # Add whether the mon is trapped or forced to switch. But first, find the index
        # index = None
        # if mon in battle.active_pokemon: index = 0 if battle.active_pokemon[0] == mon else 1
        # embeddings.append([
        #     1 if index and battle.trapped[index] else 0,
        #     1 if index and battle.force_switch[index] else 0,
        # ])

        # Flatten all the lists into a Nx1 list
        return [item for sublist in embeddings for item in sublist]

    def _describe_opp_mon_embedding(self):
        """
        Things embedded for a 173 total
            4x moves (30 each, 120 total)
            is active, current hp, fainted, is dynamaxed (1 each, 4 total)
            base stat values (6 total) (actual not known)
            status (7 total)
            types (36 total)
        """
        low_move, high_move = self._describe_move_embedding()

        low_high_dict = {'moves': {'low': low_move, 'high': high_move, 'times': 4},
                         'active': {'low': [0], 'high': [1], 'times': 1},
                         'current_hp': {'low': [0], 'high': [1000], 'times': 1},
                         'fainted': {'low': [0], 'high': [1], 'times': 1},
                         'dynamaxed': {'low': [0], 'high': [1], 'times': 1},
                         'stat_val': {'low': [0], 'high': [1000], 'times': 6},
                         'status': {'low': [0], 'high': [1], 'times': 7},
                         'type1': {'low': [0], 'high': [1], 'times': 18},
                         'type2': {'low': [0], 'high': [1], 'times': 18},
                         }

        low_opp_mon  = [sub_dict['low']  for sub_dict in low_high_dict.values() for _ in range(sub_dict['times'])]
        low_opp_mon = [item for sublist in low_opp_mon for item in sublist]

        high_opp_mon = [sub_dict['high'] for sub_dict in low_high_dict.values() for _ in range(sub_dict['times'])]
        high_opp_mon = [item for sublist in high_opp_mon for item in sublist]

        return low_opp_mon, high_opp_mon

    # Embeds the state of the battle in a X-dimensional embedding
    # Embed mons (and whether they're active)
    # Embed opponent mons (and whether they're active, they've been brought or we don't know)
    # Then embed all the Fields, Side Conditions, Weathers, Player Ratings, # of Turns and the bias
    def embed_battle(self, battle):
        """
        Things embedded for a 2113 total
            player team mons (172 each, 1032 total)
            opponent team mons (173 each, 1038 total)
            dynamax turns left per side (2 total)
            field effects (13 total)
            side conditions (20 total)
            weather (8 total)
        """
        embeddings = []

        # Add team to embeddings
        for mon in battle.team.values():
            embeddings.append(self._embed_mon(battle, mon))

        # TODO: Add embedding for current mon's boosts

        # Embed opponent's mons. teampreview_opponent_team has empty move slots while opponent_team has moves we remember.
        # We first embed opponent_active_pokemon, then ones we remember from the team, then the rest
        embedded_opp_mons = set()

        for mon in battle.opponent_team.values():
            if mon.species in embedded_opp_mons: continue
            embeddings.append(self._embed_opp_mon(battle, mon))
            embedded_opp_mons.add(mon.species)

        for mon in battle.teampreview_opponent_team:
            if mon in embedded_opp_mons: continue
            # handle multiple indifferentiable forms (i.e. 'urshifu' in team preview but 'urshifurapidstrike' once seen)
            if any(mon in seen_mon for seen_mon in embedded_opp_mons): continue
            embeddings.append(self._embed_opp_mon(battle, battle.teampreview_opponent_team[mon]))
            embedded_opp_mons.add(mon)

        # TODO: Add embedding for current opponent's mon's boosts

        # embedded_opp_mons = set()
        # for mon in battle.opponent_active_pokemon:
        #     if mon:
        #         embeddings.append(self._embed_opp_mon(battle, mon))
        #         embedded_opp_mons.add(mon.species)
        #
        # for mon in battle.opponent_team.values():
        #     if mon.species in embedded_opp_mons: continue
        #     embeddings.append(self._embed_opp_mon(battle, mon))
        #     embedded_opp_mons.add(mon.species)
        #
        # for mon in battle.teampreview_opponent_team:
        #     if mon in embedded_opp_mons: continue
        #     embeddings.append(self._embed_opp_mon(battle, battle.teampreview_opponent_team[mon]))
        #     embedded_opp_mons.add(mon)

        # Add Dynamax stuff
        # embeddings.append(battle.can_dynamax + battle.opponent_can_dynamax + [battle.dynamax_turns_left, battle.opponent_dynamax_turns_left])

        embeddings.append([
            battle.dynamax_turns_left,
            battle.opponent_dynamax_turns_left
        ])

        # Add Fields;
        embeddings.append([1 if field in battle.fields else 0 for field in self._knowledge['Field']])

        # Add Side Conditions
        embeddings.append([1 if sc in battle.side_conditions else 0 for sc in self._knowledge['SideCondition']])

        # Add Weathers
        embeddings.append([1 if weather == battle.weather else 0 for weather in self._knowledge['Weather']])

        # Add Player Ratings, the battle's turn and a bias term
        # embeddings.append(list(map(lambda x: x if x else -1, [battle.rating,
        #                                                       battle.opponent_rating,
        #                                                       battle.turn,
        #                                                       1]
        #                            )))

        # Flatten all the lists into a list
        return np.array([item for sublist in embeddings for item in sublist])

    def _describe_battle_embedding(self):
        """
        Things embedded for a 2113 total
            player team mons (172 each, 1032 total)
            opponent team mons (173 each, 1038 total)
            dynamax turns left per side (2 total)
            field effects (13 total)
            side conditions (20 total)
            weather (8 total)
        """
        low_mon, high_mon = self._describe_mon_embedding()
        low_opp_mon, high_opp_mon = self._describe_opp_mon_embedding()

        low_high_dict = {'team': {'low': low_mon, 'high': high_mon, 'times': 6},
                         'opp_team': {'low': low_opp_mon, 'high': high_opp_mon, 'times': 6},
                         'max_turns': {'low': [0], 'high': [3], 'times': 1},
                         'opp_max_turns': {'low': [0], 'high': [3], 'times': 1},
                         'field': {'low': [0], 'high': [1], 'times': 13},
                         'side_condition': {'low': [0], 'high': [1], 'times': 20},
                         'weather': {'low': [0], 'high': [1], 'times': 8},
                         }

        low_battle  = [sub_dict['low']  for sub_dict in low_high_dict.values() for _ in range(sub_dict['times'])]
        low_battle = [item for sublist in low_battle for item in sublist]

        high_battle = [sub_dict['high'] for sub_dict in low_high_dict.values() for _ in range(sub_dict['times'])]
        high_battle = [item for sublist in high_battle for item in sublist]

        return low_battle, high_battle

    def describe_embedding(self) -> Space:

        low, high = self._describe_battle_embedding()
        return Box(
            np.array(low, dtype=np.float32),
            np.array(high, dtype=np.float32),
            dtype=np.float32,
        )

    def embedding_shape(self) -> Tuple[int, int]:
        return (1, self.BATTLE_LEN)
