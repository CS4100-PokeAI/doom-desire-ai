from typing import Tuple, TypeVar

import numpy as np
from gym import Space
from gym.spaces import Box

from doom_desire.embed.abstract_embedder import AbstractFlatEmbedder
from poke_env.data import GEN_TO_MOVES, GEN_TO_POKEDEX
from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.environment.field import Field
from poke_env.environment.move import Move
from poke_env.environment.move_category import MoveCategory
from poke_env.environment.pokemon import Pokemon
from poke_env.environment.pokemon_type import PokemonType
from poke_env.environment.side_condition import SideCondition
from poke_env.environment.status import Status
from poke_env.environment.target_type import TargetType
from poke_env.environment.volatile_status import VolatileStatus
from poke_env.environment.weather import Weather

ObservationType = TypeVar("ObservationType")


class CustomTinyEmbedder(AbstractFlatEmbedder):

    def __init__(self, gen=8, priority=0):

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
        self.FIELD_LEN = 43
        self.BATTLE_LEN = 345

        self._embedding_description = self._describe_embedding()

    # Returns an array of an embedded move; could be precomputed (total length of 30 with all shortenings)
    def _embed_move(self, move: Move) -> ObservationType:
        """
        Things embedded for a 30 total
            accuracy, base power, priority (1 each, 3 total)
            move category OHE (3 total)
            move type OHE (18 total)
            stat boosts (5 total)
            chance of secondary (1 total)
        """

        # If the move is None or empty, return a negative array (filled w/ -1's)
        if move is None or move.is_empty:
            return [-1] * self.MOVE_LEN

        embeddings = []

        # Encode other properties
        embeddings.append([  # Total length of 3
            move.accuracy,
            move.base_power / 100,  # maybe normalizing this will help
            move.priority,
        ])

        # OHE Move, Category, Defensive Category, Move Type
        embeddings.append([1 if move.category == category else 0 for category in self._knowledge['MoveCategory']])

        embeddings.append([1 if move.type == pokemon_type else 0 for pokemon_type in self._knowledge['PokemonType']])

        # OHE Fields, SC, Weather (bad coding -- assumes field name will be move name, and uses string manipulation)

        # OHE Targeting Types

        # OHE Volatility Statuses

        # OHE Statuses

        # OHE Boosts to the move's target (which sometimes are self-boosts)
        boost_embeddings = {'atk': 0, 'def': 0, 'spa': 0, 'spd': 0, 'spe': 0, 'evasion': 0, 'accuracy': 0}
        if move.boosts:
            for stat in move.boosts: boost_embeddings[stat] += move.boosts[stat]
        elif move.secondary:
            for x in move.secondary:
                for stat in x.get('boosts', {}): boost_embeddings[stat] += x['boosts'][stat]
        embeddings.append(list(boost_embeddings.values())[:-2])  # ignore evasion and accuracy boosts

        # Add Self-Boosts

        # Introduce the chance of a secondary effect happening, total length of only 1
        chance = 0
        for x in move.secondary:
            chance = max(chance, x.get('chance', 0))
        embeddings.append([chance / 100])  # normalizing

        # Flatten the arrays
        return [item for sublist in embeddings for item in sublist]

    def _describe_move_embedding(self) -> Space:
        """
        Things embedded for a 30 total
            accuracy, base power, priority (1 each, 3 total)
            move category OHE (3 total)
            move type OHE (18 total)
            stat boosts (5 total)
            chance of secondary (1 total)
        """
        low_high_dict = {'accuracy': {'low': -1, 'high': 1, 'times': 1},
                         'base_power': {'low': -1, 'high': 3, 'times': 1},
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
    def _embed_mon(self, mon: Pokemon) -> ObservationType:
        """
        Things embedded for a 172 total
            4x moves (30 each, 120 total)
            is active, current hp, fainted, is dynamaxed (1 each, 4 total)
            stat values (5 total) (doesn't include HP)
            status (7 total)
            types (36 total)
        """
        embeddings = []

        # Append moves to embedding (and account for the fact that the mon might have <4 moves)
        for move in (list(mon.moves.values()) + [None, None, None, None])[:4]:
            embeddings.append(self._embed_move(move))

        # Add whether the mon is active, the current hp, whether its fainted, its level,
        # its weight and whether its recharging or preparing
        embeddings.append([
            int(mon.active),
            mon.current_hp / 100,  # normalizing
            int(mon.fainted),
            int(mon.is_dynamaxed),
        ])

        # Add stats and boosts
        embeddings.append(stat / 100 for stat in mon.stats.values())
        # embeddings.append(mon.boosts.values())  # only the current can have boosts anyway so don't do for all

        # Add status (one-hot encoded)
        embeddings.append([1 if mon.status == status else 0 for status in self._knowledge['Status']])

        # Add Types (one-hot encoded)
        embeddings.append([1 if mon.type_1 == pokemon_type else 0 for pokemon_type in self._knowledge['PokemonType']])
        embeddings.append([1 if mon.type_2 == pokemon_type else 0 for pokemon_type in self._knowledge['PokemonType']])

        # Flatten all the lists into a Nx1 list
        return [item for sublist in embeddings for item in sublist]

    def _describe_mon_embedding(self) -> Space:
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
                         'current_hp': {'low': [0], 'high': [10], 'times': 1},
                         'fainted': {'low': [0], 'high': [1], 'times': 1},
                         'dynamaxed': {'low': [0], 'high': [1], 'times': 1},
                         'stat_val': {'low': [0], 'high': [10], 'times': 5},
                         'status': {'low': [0], 'high': [1], 'times': 7},
                         'type1': {'low': [0], 'high': [1], 'times': 18},
                         'type2': {'low': [0], 'high': [1], 'times': 18},
                         }

        low_mon  = [sub_dict['low']  for sub_dict in low_high_dict.values() for _ in range(sub_dict['times'])]
        low_mon = [item for sublist in low_mon for item in sublist]

        high_mon = [sub_dict['high'] for sub_dict in low_high_dict.values() for _ in range(sub_dict['times'])]
        high_mon = [item for sublist in high_mon for item in sublist]

        return low_mon, high_mon

    def _embed_opp_mon(self, mon: Pokemon) -> ObservationType:
        """
        Things embedded for a 173 total
            4x moves (30 each, 120 total)
            is active, current hp, fainted, is dynamaxed (1 each, 4 total)
            base stat values (6 total) (actual not known)
            status (7 total)
            types (36 total)
        """
        embeddings = []

        # Append moves to embedding (and account for the fact that the mon might have <4 moves, or we don't know of them)
        for move in (list(mon.moves.values()) + [None, None, None, None])[:4]:
            embeddings.append(self._embed_move(move))

        # Add whether the mon is active, the current hp, whether its fainted, its level, its weight and whether its recharging or preparing
        embeddings.append([
            int(mon.active),  # This mon is on the field now
            mon.current_hp / 100,
            int(mon.fainted),
            int(mon.is_dynamaxed),
        ])

        # Add stats and boosts
        embeddings.append(stat / 100 for stat in mon.base_stats.values())
        # embeddings.append(mon.boosts.values())

        # Add status (one-hot encoded)
        embeddings.append([1 if mon.status == status else 0 for status in self._knowledge['Status']])

        # Add Types (one-hot encoded)
        embeddings.append([1 if mon.type_1 == pokemon_type else 0 for pokemon_type in self._knowledge['PokemonType']])
        embeddings.append([1 if mon.type_2 == pokemon_type else 0 for pokemon_type in self._knowledge['PokemonType']])

        # Flatten all the lists into a Nx1 list
        return [item for sublist in embeddings for item in sublist]

    def _describe_opp_mon_embedding(self) -> Space:
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
                         'current_hp': {'low': [0], 'high': [10], 'times': 1},
                         'fainted': {'low': [0], 'high': [1], 'times': 1},
                         'dynamaxed': {'low': [0], 'high': [1], 'times': 1},
                         'base_stat': {'low': [0], 'high': [10], 'times': 6},
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
    def embed_battle(self, battle: AbstractBattle) -> ObservationType:
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
        embeddings.append(self._embed_mon(battle.active_pokemon))
        embeddings.append(self._embed_opp_mon(battle.opponent_active_pokemon))

        return_embedding = np.float32([item for sublist in embeddings for item in sublist])

        if any(return_embedding[i] < self._embedding_description.low[i] for i in range(len(self._embedding_description.low))):
            print("Embedding value lower than limit: \n", return_embedding, self._embedding_description.low)
        for i in range(len(self._embedding_description.high)):
            if return_embedding[i] > self._embedding_description.high[i]:
                print("Embedding value higher than limit (i=", i,") : ", return_embedding[i], " > ", self._embedding_description.high[i])

        return return_embedding

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

        low_high_dict = {'team': {'low': low_mon, 'high': high_mon, 'times': 1},
                         'opp_team': {'low': low_opp_mon, 'high': high_opp_mon, 'times': 1},
                         }

        low_battle  = [sub_dict['low']  for sub_dict in low_high_dict.values() for _ in range(sub_dict['times'])]
        low_battle = [item for sublist in low_battle for item in sublist]

        high_battle = [sub_dict['high'] for sub_dict in low_high_dict.values() for _ in range(sub_dict['times'])]
        high_battle = [item for sublist in high_battle for item in sublist]

        return low_battle, high_battle

    def _describe_embedding(self) -> Box:

        low, high = self._describe_battle_embedding()
        return Box(
            np.array(low, dtype=np.float32),
            np.array(high, dtype=np.float32),
            dtype=np.float32,
        )

    def describe_embedding(self) -> Space:
        return self._embedding_description

    def embedding_shape(self) -> Tuple[int, int]:
        return (1, self.BATTLE_LEN)
