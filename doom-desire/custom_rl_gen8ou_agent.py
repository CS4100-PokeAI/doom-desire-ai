# -*- coding: utf-8 -*-
import asyncio
from types import SimpleNamespace

import wandb
from tabulate import tabulate

from doom_desire.embed.custom_embedder import CustomEmbedder
from doom_desire.embed.custom_tiny_embedder import CustomTinyEmbedder
from doom_desire.embed.matchup_embedder import MatchupEmbedder
from doom_desire.embed.simple_plus_embedder import SimplePlusEmbedder
from doom_desire.example_teams.gen8ou import RandomTeamFromPool, team_1, team_2
from doom_desire.embed.simple_embedder import SimpleEmbedder
from doom_desire.example_teams.team_repo import TeamRepository
from doom_desire.helpers.reward_calculator import RewardCalculator, CustomRewardCalculator, ExampleRewardCalculator
from doom_desire.models.battle_modeler import SequentialBattleModeler
from doom_desire.player.custom_player import CustomRLPlayer
from poke_env.player.baselines import MaxBasePowerPlayer, SimpleHeuristicsPlayer
from poke_env.player.random_player import RandomPlayer

from poke_env.player.utils import background_evaluate_player, background_cross_evaluate
from poke_env.teambuilder.teambuilder import Teambuilder


run_config_dict = {
    'train': True,
    'evaluate': True,
    'cross_evaluate': True,
    'evaluation_battles': 100,

    'embedder': 'matchup',
    'reward_calculator': 'example',

    'team': TeamRepository.ou_teams_as_list,
    'opponent': 'rand',
    'in_order': True,
    'opponent_team': TeamRepository.ou_teams_as_list,

    'load_weights': False,
    'weights_file': 'model_matchup_rand200k.hdf5',
    'force_save_model': True,
}
run_config = SimpleNamespace(**run_config_dict)


model_config_dict = {
    'NB_TRAINING_STEPS': 200000,
    'NB_EVALUATION_EPISODES': 100,
    'first_layer_nodes': 256,  # 128       or 500
    'second_layer_nodes': 128,  # 64        or 500
    'third_layer_nodes': -1,
    'gamma': 0.8,  # 0.5       or 0.99
    'delta_clip': .9,  # .01       or 0.9
    'target_model_update': 1,  # 1         or 10
    'learning_rate': .00025,  # 0.00025   or 0.001
    'memory_limit': 100000,
    'warmup_steps': 1000,  # 1000      or 500
    'activation': "relu",  # "elu"     or "relu"
    'policy': 'EpsGreedyQPolicy',
}
model_config = SimpleNamespace(**model_config_dict)


embedders = {
    'simple': SimpleEmbedder(),
    'custom': CustomEmbedder(),
    'custom_tiny': CustomTinyEmbedder(),
    'matchup': MatchupEmbedder(),
    'simple_plus': SimplePlusEmbedder()
}

reward_calculators = {
    'example': ExampleRewardCalculator,
    'custom': CustomRewardCalculator,
}


def set_training_opponents(config):
    opponent_teams = RandomTeamFromPool(config.opponent_team)

    training_opponent = None
    if config.opponent == 'rand':
        training_opponent = [RandomPlayer(battle_format="gen8ou", team=opponent_teams)]
    elif config.opponent == 'max':
        training_opponent = [MaxBasePowerPlayer(battle_format="gen8ou", team=opponent_teams)]
    elif config.opponent == 'heuristic':
        training_opponent = [SimpleHeuristicsPlayer(battle_format="gen8ou", team=opponent_teams)]
    elif config.opponent == 'rand-max':
        # Each battle is going to be against either a random player or a max base power player
        training_opponent = [RandomPlayer(battle_format="gen8ou", team=opponent_teams),
                             MaxBasePowerPlayer(battle_format="gen8ou", team=opponent_teams)]
    elif config.opponent == 'rand-heuristic':
        # Each battle is going to be against either a random player or heuristic player
        training_opponent = [RandomPlayer(battle_format="gen8ou", team=opponent_teams),
                             SimpleHeuristicsPlayer(battle_format="gen8ou", team=opponent_teams)]
    elif config.opponent == 'max-heuristic':
        # Each battle is going to be against either max base power player or heuristic player
        training_opponent = [MaxBasePowerPlayer(battle_format="gen8ou", team=opponent_teams),
                             SimpleHeuristicsPlayer(battle_format="gen8ou", team=opponent_teams)]
    elif config.opponent == 'all':
        # Each battle is going to be one of the players
        training_opponent = [RandomPlayer(battle_format="gen8ou", team=opponent_teams),
                             MaxBasePowerPlayer(battle_format="gen8ou", team=opponent_teams),
                             SimpleHeuristicsPlayer(battle_format="gen8ou", team=opponent_teams)]
    elif config.opponent == 'combo':
        # Each battle is going to be one of the players
        training_opponent = [RandomPlayer(battle_format="gen8ou", team=opponent_teams),
                             RandomPlayer(battle_format="gen8ou", team=opponent_teams),
                             # MaxBasePowerPlayer(battle_format="gen8ou", team=opponent_teams),
                             # [RandomPlayer(battle_format="gen8ou", team=opponent_teams),
                             #  MaxBasePowerPlayer(battle_format="gen8ou", team=opponent_teams)]
                             #  MaxBasePowerPlayer(battle_format="gen8ou", team=opponent_teams),
                             #  SimpleHeuristicsPlayer(battle_format="gen8ou", team=opponent_teams)]
                             # SimpleHeuristicsPlayer(battle_format="gen8ou", team=opponent_teams),
                             # [RandomPlayer(battle_format="gen8ou", team=opponent_teams),
                             #  MaxBasePowerPlayer(battle_format="gen8ou", team=opponent_teams),
                             #  SimpleHeuristicsPlayer(battle_format="gen8ou", team=opponent_teams)]
                             ]

    return training_opponent


def evaluate_trained_agent(config, trained_agent: CustomRLPlayer):
    opponent_teams = RandomTeamFromPool(run_config.opponent_team)

    # Create the first opponent to evaluate against
    eval_opponent = RandomPlayer(battle_format="gen8ou", team=opponent_teams)
    print("Results against random player:")
    trained_agent.evaluate_model(opponent=eval_opponent, num_battles=config.evaluation_battles, verbose_end=True)

    second_opponent = MaxBasePowerPlayer(battle_format="gen8ou", team=opponent_teams)
    print("Results against max base power player:")
    trained_agent.evaluate_model(opponent=second_opponent, num_battles=config.evaluation_battles, verbose_end=True)

    third_opponent = SimpleHeuristicsPlayer(battle_format="gen8ou", team=opponent_teams)
    print("Results against simple heuristics player:")
    trained_agent.evaluate_model(opponent=third_opponent, num_battles=config.evaluation_battles, verbose_end=True)


async def cross_evaluate_trained_agent(config, trained_agent: CustomRLPlayer):
    # Evaluating the model
    opponent_teams = RandomTeamFromPool(config.opponent_team)

    # random_opponent = RandomPlayer(battle_format="gen8ou", team=opponent_teams)
    if trained_agent.challenge_task:
        trained_agent.reset_env(restart=False)

    # Evaluate the player with included util method
    n_challenges = 250
    placement_battles = 40
    eval_task = background_evaluate_player(
        trained_agent.agent, n_challenges, placement_battles
    )
    trained_agent.test(num_battles=n_challenges)
    print("Evaluation with included method:", eval_task.result())
    trained_agent.reset_env(restart=False)

    # Cross evaluate the player with included util method
    n_challenges = 50

    players = [
        trained_agent.agent,
        RandomPlayer(battle_format="gen8ou", team=opponent_teams),
        MaxBasePowerPlayer(battle_format="gen8ou", team=opponent_teams),
        SimpleHeuristicsPlayer(battle_format="gen8ou", team=opponent_teams),
    ]
    cross_eval_task = background_cross_evaluate(players, n_challenges)
    trained_agent.test(num_battles=n_challenges * (len(players) - 1))
    cross_evaluation = cross_eval_task.result()
    table = [["-"] + [p.username for p in players]]
    for p_1, results in cross_evaluation.items():
        table.append([p_1] + [cross_evaluation[p_1][p_2] for p_2 in results])
    print("Cross evaluation of DQN with baselines:")
    print(tabulate(table))


async def main():
    # Initialize a new wandb run; We can use os.environ['WANDB_MODE'] = 'dryrun' to not save wandb to cloud
    if run_config.train:
        wandb.init(config=model_config_dict, entity="jfenton888", project="doom-desire-ai_DQN")
        config = wandb.config

    # Create one environment for both training and evaluation
    player_teams = RandomTeamFromPool(run_config.team)
    training_agent = CustomRLPlayer(battle_format="gen8ou",
                                    battle_modeler=SequentialBattleModeler(model_config,
                                                                           embedder=embedders[run_config.embedder]
                                                                           ),
                                    reward_calculator=reward_calculators[run_config.reward_calculator],
                                    team=player_teams,
                                    start_challenging=False,

                                    build_model=True,
                                    use_wandb=True,
                                    )
    if run_config.load_weights:
        training_agent.load_model(run_config.weights_file)

    # training_agent.visualize_model()  # TODO: this doesn't work

    if run_config.train:
        # Train the agent against the opponent for num_steps
        training_opponent = set_training_opponents(run_config)
        training_agent.train(training_opponent,
                             num_steps=model_config.NB_TRAINING_STEPS,
                             in_order=run_config.in_order)

    if run_config.evaluate:
        evaluate_trained_agent(run_config, trained_agent=training_agent)

    if run_config.cross_evaluate:
        await cross_evaluate_trained_agent(run_config, trained_agent=training_agent)

    if run_config.force_save_model or input("Save the model? (y/n): ") is 'y':
        training_agent.save_model()

    training_agent.close()


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())
