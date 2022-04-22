# -*- coding: utf-8 -*-
import asyncio
import sys
import random

from doom_desire.embed.custom_embedder import CustomEmbedder
from doom_desire.example_teams.gen8ou import team_1, team_2
from doom_desire.player.custom_player import CustomRLPlayer
from poke_env.player.random_player import RandomPlayer
from poke_env.player.player import Player
from poke_env.player.utils import cross_evaluate
from tabulate import tabulate

class TempRandomPlayer(Player):
    # _plyr = CustomRLPlayer()
    _custom_embedder = CustomEmbedder()

    def choose_move(self, battle):
        order = self.choose_default_move()

        print("Battle Turn: ", battle.turn)
        print("    Len of Move Embeddings: ", len(self._custom_embedder._embed_move(list(battle.active_pokemon.moves.values())[0])))
        print("    Len of Mon Embeddings: ", len(self._custom_embedder._embed_mon(battle, battle.active_pokemon)))
        print("    Len of Opponent Mon Embeddings: ", len(self._custom_embedder._embed_opp_mon(battle, list(battle.opponent_team.values())[0])))
        print("    Len of Battle Embeddings: ", len(self._custom_embedder.embed_battle(battle)))

        # if battle.turn == 1 and battle.active_pokemon[0]:
        #     print()
        #     print("Len of Move Embeddings: ", len(self._plyr._embed_move(list(battle.active_pokemon[0].moves.values())[0])))
        #     print("Len of Mon Embeddings: ", len(self._plyr._embed_mon(battle, battle.active_pokemon[0])))
        #     print("Len of Opponent Mon Embeddings: ", len(self._plyr._embed_opp_mon(battle, battle.opponent_active_pokemon[0])))
        #     print("Len of Battle Embeddings: ", len(self._plyr.embed_battle(battle)))

        return order

    def teampreview(self, battle):

        # We use 1-6 because  showdown's indexes start from 1
        return "/team " + "".join(random.sample(list(map(lambda x: str(x+1), range(0, len(battle.team)))), k=4))


# To run from command line, run this in the root directory: python3.8 simulators/simulate_random_doubles.py
async def main():
    print("\033[92m Starting script... \033[0m")

    # We create players:
    players = [
      RandomPlayer(max_concurrent_battles=1, battle_format='gen8ou', team=team_1),
      TempRandomPlayer(max_concurrent_battles=1, battle_format='gen8ou', team=team_1),
    ]

    # Each player plays n times against eac other
    n = 10

    # Pit players against each other
    print("About to start " + str(n*sum(i for i in range(0, len(players)))) + " battles...")
    cross_evaluation = await cross_evaluate(players, n_challenges=n)

    # Defines a header for displaying results
    table = [["-"] + [p.username for p in players]]

    # Adds one line per player with corresponding results
    for p_1, results in cross_evaluation.items():
        table.append([p_1] + [cross_evaluation[p_1][p_2] for p_2 in results])

    # Displays results in a nicely formatted table.
    print(tabulate(table))


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())
