# -*- coding: utf-8 -*-
import asyncio

import numpy as np
import tensorflow as tf
from gym.utils.env_checker import check_env
from tabulate import tabulate

from doom_desire.env_algorithm.dnq_training import example_dqn_structure
from doom_desire.env_algorithm.dqn_evaluation import example_dqn_evaluation
from doom_desire.example_teams.gen8ou import RandomTeamFromPool, team_1, team_2
from doom_desire.models.model_builder import ExampleSequentialModelBuilder
from doom_desire.player.rl_player_examples import ExampleRLPlayer
from poke_env.player.baselines import MaxBasePowerPlayer, SimpleHeuristicsPlayer
from poke_env.player.random_player import RandomPlayer

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from tensorflow.keras.optimizers import Adam

from poke_env.player.utils import background_evaluate_player, background_cross_evaluate


async def main():
    # custom_builder = RandomTeamFromPool([team_1, team_2])
    custom_builder = RandomTeamFromPool([team_1])

    # Create one environment for training and one for evaluation
    # opponent = SimpleHeuristicsPlayer(battle_format="gen8ou", team=custom_builder)
    train_opponent = RandomPlayer(battle_format="gen8ou", team=custom_builder)

    train_env = ExampleRLPlayer(
        battle_format="gen8ou", team=custom_builder, opponent=train_opponent, start_challenging=True
    )
    eval_opponent = RandomPlayer(battle_format="gen8ou", team=custom_builder)
    eval_env = ExampleRLPlayer(
        battle_format="gen8ou", team=custom_builder, opponent=eval_opponent, start_challenging=True
    )

    # Compute dimensions
    input_shape = (1,) + train_env.observation_space.shape
    action_space_size = train_env.action_space.n

    # Create model
    model_builder = ExampleSequentialModelBuilder()
    model = model_builder.build_shaped_model(input_shape=input_shape, output_size=action_space_size)
    # model = Sequential()
    # model.add(Dense(128, activation="elu", input_shape=input_shape))
    # model.add(Flatten())
    # model.add(Dense(64, activation="elu"))
    # model.add(Dense(n_action, activation="linear"))

    # Defining the DQN
    memory = SequentialMemory(limit=10000, window_length=1)
    dqn = example_dqn_structure(model=model, nb_actions=action_space_size, memory=memory)

    # Training the model
    dqn.fit(train_env, nb_steps=20000)
    train_env.close()


    # Evaluating the model
    print("Results against random player:")
    dqn.test(eval_env, nb_episodes=100, verbose=False, visualize=False)
    print(
        f"DQN Evaluation: {eval_env.n_won_battles} victories out of {eval_env.n_finished_battles} episodes"
    )

    second_opponent = MaxBasePowerPlayer(battle_format="gen8ou", team=custom_builder)
    eval_env.reset_env(restart=True, opponent=second_opponent)
    print("Results against max base power player:")
    dqn.test(eval_env, nb_episodes=100, verbose=False, visualize=False)
    print(
        f"DQN Evaluation: {eval_env.n_won_battles} victories out of {eval_env.n_finished_battles} episodes"
    )

    third_opponent = SimpleHeuristicsPlayer(battle_format="gen8ou", team=custom_builder)
    eval_env.reset_env(restart=True, opponent=third_opponent)
    print("Results against simple heuristics player:")
    dqn.test(eval_env, nb_episodes=100, verbose=False, visualize=False)
    print(
        f"DQN Evaluation: {eval_env.n_won_battles} victories out of {eval_env.n_finished_battles} episodes"
    )

    eval_env.close()

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

