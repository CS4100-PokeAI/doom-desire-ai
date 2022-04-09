# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

from doom_desire.env_algorithm.dnq_training import example_dqn_training
from doom_desire.env_algorithm.dqn_evaluation import example_dqn_evaluation
from doom_desire.models.model_builder import ExampleSequentialModelBuilder
from doom_desire.player.rl_player_examples import ExampleRLPlayer, MaxDamagePlayer
from poke_env.player.random_player import RandomPlayer

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from tensorflow.keras.optimizers import Adam


NB_TRAINING_STEPS = 10000
NB_EVALUATION_EPISODES = 100

tf.random.set_seed(0)
np.random.seed(0)



if __name__ == "__main__":
    env_player = ExampleRLPlayer(battle_format="gen8randombattle")

    opponent = RandomPlayer(battle_format="gen8randombattle")
    second_opponent = MaxDamagePlayer(battle_format="gen8randombattle")

    model_builder = ExampleSequentialModelBuilder()
    model = model_builder.build_shaped_model(input_shape=(1, 10), output_size=len(env_player.action_space))

    memory = SequentialMemory(limit=10000, window_length=1)

    # Simple epsilon greedy
    policy = LinearAnnealedPolicy(
        EpsGreedyQPolicy(),
        attr="eps",
        value_max=1.0,
        value_min=0.05,
        value_test=0,
        nb_steps=10000,
    )

    # Defining our DQN
    dqn = DQNAgent(
        model=model,
        nb_actions=len(env_player.action_space),
        policy=policy,
        memory=memory,
        nb_steps_warmup=1000,
        gamma=0.5,
        target_model_update=1,
        delta_clip=0.01,
        enable_double_dqn=True,
    )

    dqn.compile(Adam(lr=0.00025), metrics=["mae"])

    # Training
    env_player.play_against(
        env_algorithm=example_dqn_training,
        opponent=opponent,
        env_algorithm_kwargs={"dqn": dqn, "nb_steps": NB_TRAINING_STEPS},
    )
    model.save("model_%d" % NB_TRAINING_STEPS)

    # Evaluation
    print("Results against random player:")
    env_player.play_against(
        env_algorithm=example_dqn_evaluation,
        opponent=opponent,
        env_algorithm_kwargs={"dqn": dqn, "nb_episodes": NB_EVALUATION_EPISODES},
    )

    print("\nResults against max player:")
    env_player.play_against(
        env_algorithm=example_dqn_evaluation,
        opponent=second_opponent,
        env_algorithm_kwargs={"dqn": dqn, "nb_episodes": NB_EVALUATION_EPISODES},
    )
