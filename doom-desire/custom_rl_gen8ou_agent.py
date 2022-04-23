# -*- coding: utf-8 -*-
import asyncio

import wandb

from doom_desire.embed.custom_embedder import CustomEmbedder
from doom_desire.example_teams.gen8ou import RandomTeamFromPool, team_1, team_2
from doom_desire.embed.simple_embedder import SimpleEmbedder
from doom_desire.helpers.reward_calculator import RewardCalculator
from doom_desire.player.custom_player import CustomRLPlayer
from poke_env.player.baselines import MaxBasePowerPlayer, SimpleHeuristicsPlayer
from poke_env.player.random_player import RandomPlayer


# Defaults taken from reuniclusVGC
# config_defaults = {
#     'NB_TRAINING_STEPS': 10000,
#     'NB_EVALUATION_EPISODES': 100,
#     'first_layer_nodes': 500,
#     'second_layer_nodes': 500,
#     'third_layer_nodes': -1,
#     'gamma': .99,
#     'delta_clip': .9,
#     'target_model_update': 10,
#     'lr': .001,
#     'memory_limit': 100000,
#     'warmup_steps': 500,
#     'activation': "relu",
#     'policy': 'EpsGreedyQPolicy',
#     'team': 'swampert',
#     'opponent': 'max',
#     'opponent_team': 'swampert'
# }

async def main():


    config_defaults = {
        'NB_TRAINING_STEPS': 10000,
        'NB_EVALUATION_EPISODES': 100,
        'first_layer_nodes': 256,   # 128       or 500
        'second_layer_nodes': 128,   # 64        or 500
        'third_layer_nodes': -1,
        'gamma': 0.8,               # 0.5       or 0.99
        'delta_clip': .9,          # .01       or 0.9
        'target_model_update': 1,   # 1         or 10
        'learning_rate': .001,    # 0.00025   or 0.001
        'memory_limit': 100000,
        'warmup_steps': 1000,       # 1000      or 500
        'activation': "relu",        # "elu"     or "relu"
        'policy': 'EpsGreedyQPolicy',
        'team': 'swampert',
        'opponent': 'rand',
        'opponent_team': 'swampert',
        'load_weights': 'model_2022_04_22_21_16_15.hdf5'  # None
    }

    # Initialize a new wandb run; We can use os.environ['WANDB_MODE'] = 'dryrun' to not save wandb to cloud
    wandb.init(config=config_defaults, entity="jfenton888", project="doom-desire-ai_DQN")
    config = wandb.config

    # custom_builder = RandomTeamFromPool([team_1, team_2])
    custom_builder = RandomTeamFromPool([team_1, team_2])

    training_opponent = None
    if config.opponent == 'rand':
        training_opponent = [RandomPlayer(battle_format="gen8ou", team=custom_builder)]
    elif config.opponent == 'max':
        training_opponent = [MaxBasePowerPlayer(battle_format="gen8ou", team=custom_builder)]
    elif config.opponent == 'heuristic':
        training_opponent = [SimpleHeuristicsPlayer(battle_format="gen8ou", team=custom_builder)]
    elif config.opponent == 'rand-max':
        # Each battle is going to be against either a random player or a max base power player
        training_opponent = [RandomPlayer(battle_format="gen8ou", team=custom_builder),
                             MaxBasePowerPlayer(battle_format="gen8ou", team=custom_builder)]
    elif config.opponent == 'rand-heuristic':
        # Each battle is going to be against either a random player or heuristic player
        training_opponent = [RandomPlayer(battle_format="gen8ou", team=custom_builder),
                             SimpleHeuristicsPlayer(battle_format="gen8ou", team=custom_builder)]
    elif config.opponent == 'max-heuristic':
        # Each battle is going to be against either max base power player or heuristic player
        training_opponent = [MaxBasePowerPlayer(battle_format="gen8ou", team=custom_builder),
                             SimpleHeuristicsPlayer(battle_format="gen8ou", team=custom_builder)]
    elif config.opponent == 'all':
        # Each battle is going to be one of the players
        training_opponent = [RandomPlayer(battle_format="gen8ou", team=custom_builder),
                             MaxBasePowerPlayer(battle_format="gen8ou", team=custom_builder),
                             SimpleHeuristicsPlayer(battle_format="gen8ou", team=custom_builder)]



    # Create one environment for both training and evaluation
    training_agent = CustomRLPlayer(battle_format="gen8ou",
                                    config=config,
                                    embedder=CustomEmbedder(),
                                    reward_calculator=RewardCalculator(),
                                    team=custom_builder,
                                    start_challenging=False)
    if config.load_weights:
        training_agent.load_model(config.load_weights)

    # training_agent.visualize_model()  # TODO: this doesn't work
    # Train the agent against the opponent for num_steps
    training_agent.train(training_opponent,
                         num_steps=config.NB_TRAINING_STEPS,
                         in_order=True)


    # Create the first opponent to evaluate against
    eval_opponent = RandomPlayer(battle_format="gen8ou", team=custom_builder)
    # Reset the environment and set the new opponent to play against
    training_agent.reset_env(opponent=eval_opponent, restart=True)

    print("Results against random player:")
    training_agent.evaluate_model(num_battles=config.NB_EVALUATION_EPISODES)
    # dqn.test(eval_env, nb_episodes=100, verbose=False, visualize=False)
    print(
        f"DQN Evaluation: {training_agent.n_won_battles} victories out of {training_agent.n_finished_battles} episodes"
    )

    second_opponent = MaxBasePowerPlayer(battle_format="gen8ou", team=custom_builder)
    training_agent.reset_env(opponent=second_opponent, restart=True)
    print("Results against max base power player:")
    training_agent.evaluate_model(num_battles=config.NB_EVALUATION_EPISODES)
    print(
        f"DQN Evaluation: {training_agent.n_won_battles} victories out of {training_agent.n_finished_battles} episodes"
    )

    third_opponent = SimpleHeuristicsPlayer(battle_format="gen8ou", team=custom_builder)
    training_agent.reset_env(opponent=third_opponent, restart=True)
    print("Results against simple heuristics player:")
    training_agent.evaluate_model(num_battles=config.NB_EVALUATION_EPISODES)
    print(
        f"DQN Evaluation: {training_agent.n_won_battles} victories out of {training_agent.n_finished_battles} episodes"
    )

    if input("Save the model? (y/n): ") is 'y':
        training_agent.save_model()

    training_agent.close()





    # Evaluating the model

    # eval_env.reset_env(restart=False)

    # # Evaluate the player with included util method
    # n_challenges = 250
    # placement_battles = 40
    # eval_task = background_evaluate_player(
    #     eval_env.agent, n_challenges, placement_battles
    # )
    # dqn.test(eval_env, nb_episodes=n_challenges, verbose=False, visualize=False)
    # print("Evaluation with included method:", eval_task.result())
    # eval_env.reset_env(restart=False)
    #
    # # Cross evaluate the player with included util method
    # n_challenges = 50
    # players = [
    #     eval_env.agent,
    #     RandomPlayer(battle_format="gen8ou", team=custom_builder),
    #     MaxBasePowerPlayer(battle_format="gen8ou", team=custom_builder),
    #     SimpleHeuristicsPlayer(battle_format="gen8ou", team=custom_builder),
    # ]
    # cross_eval_task = background_cross_evaluate(players, n_challenges)
    # dqn.test(
    #     eval_env,
    #     nb_episodes=n_challenges * (len(players) - 1),
    #     verbose=False,
    #     visualize=False,
    # )
    # cross_evaluation = cross_eval_task.result()
    # table = [["-"] + [p.username for p in players]]
    # for p_1, results in cross_evaluation.items():
    #     table.append([p_1] + [cross_evaluation[p_1][p_2] for p_2 in results])
    # print("Cross evaluation of DQN with baselines:")
    # print(tabulate(table))


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())

