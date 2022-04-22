from datetime import datetime
from typing import Optional, Union, List

import numpy as np
import tensorflow as tf
import wandb as wandb
from gym import Space
from wandb.keras import WandbCallback

from doom_desire.helpers.abstract_embedder import AbstractEmbedder
from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.player.env_player import Gen8EnvSinglePlayer
from poke_env.player.openai_api import ObservationType
from poke_env.player.player import Player
from poke_env.player_configuration import PlayerConfiguration
from poke_env.server_configuration import ServerConfiguration
from poke_env.teambuilder.teambuilder import Teambuilder

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, MaxBoltzmannQPolicy, Policy, EpsGreedyQPolicy
from tensorflow.python.keras.engine import training
from rl.memory import SequentialMemory, Memory

from tensorflow.keras.layers import Dense, Flatten, Activation, BatchNormalization, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


class CustomRLPlayer(Gen8EnvSinglePlayer):

    def __init__(
            self,
            battle_format: Optional[str] = None,
            config: wandb.wandb_sdk.Config = None,
            embedder: AbstractEmbedder = None,
            *,
            player_configuration: Optional[PlayerConfiguration] = None,
            opponent: Optional[Union[Player, str]] = None,
            avatar: Optional[int] = None,
            log_level: Optional[int] = None,
            save_replays: Union[bool, str] = False,
            server_configuration: Optional[ServerConfiguration] = None,
            start_listening: bool = True,
            start_timer_on_battle_start: bool = False,
            team: Optional[Union[str, Teambuilder]] = None,
            start_challenging: bool = False,
    ):
        self._config = config

        self._embedder = embedder

        super().__init__(
            player_configuration=player_configuration,
            avatar=avatar,
            battle_format=battle_format,
            log_level=log_level,
            save_replays=save_replays,
            server_configuration=server_configuration,
            start_listening=start_listening,
            start_timer_on_battle_start=start_timer_on_battle_start,
            team=team,
            start_challenging=start_challenging,
        )

        self._create_model()

    def _create_model(self):

        self._model = Sequential()
        self._model.add(Dense(self._config.first_layer_nodes,  # 128
                              activation="elu",
                              input_shape=self._embedder.embedding_shape()))

        # Flattening resolve potential issues that would arise otherwise
        self._model.add(Flatten())
        if self._config.second_layer_nodes > 0:
            self._model.add(Dense(self._config.second_layer_nodes,  # 64
                                  activation="elu"))

        if self._config.third_layer_nodes > 0:
            self._model.add(Dense(self._config.third_layer_nodes,
                                  activation=self._config.activation,
                                  kernel_initializer='he_uniform'))

        self._model.add(Dense(units=self.action_space_size(), activation="linear"))

        self._policy = None
        if self._config.policy == 'MaxBoltzmannQPolicy':
            self._policy = MaxBoltzmannQPolicy() # https://github.com/keras-rl/keras-rl/blob/master/rl/policy.py#L242
        elif self._config.policy == 'EpsGreedyQPolicy':
            self._policy = EpsGreedyQPolicy(eps=.1)
        elif self._config.policy == 'LinearAnnealedPolicy':
            self._policy = LinearAnnealedPolicy(
                EpsGreedyQPolicy(),
                attr="eps",
                value_max=1.0,   # 1.0  or 0.5
                value_min=0.05,  # 0.05 or 0.025
                value_test=0,
                nb_steps=self._config.NB_TRAINING_STEPS,
            )

        self._memory = SequentialMemory(limit=self._config.memory_limit,
                                        window_length=1)

        # Defining our DQN
        self._dqn = DQNAgent(
            model=self._model,
            nb_actions=self.action_space_size(),
            policy=self._policy,
            memory=self._memory,
            nb_steps_warmup=self._config.warmup_steps,  # 1000
            gamma=self._config.gamma,  # 0.5
            target_model_update=self._config.target_model_update,  # 1
            delta_clip=self._config.delta_clip,  # 0.01
            enable_double_dqn=True,
        )
        self._dqn.compile(Adam(learning_rate=self._config.learning_rate),
                          metrics=["mae"])  # learning_rate=0.00025


        # Simple model where only one layer feeds into the next
        """
        self._model = Sequential()
        self._model.add(Dense(self._config.first_layer_nodes,
                              activation=self._config.activation,
                              input_shape=self._embedder.embedding_shape(),
                              kernel_initializer='he_uniform'))
        self._model.add(Flatten())
        if self._config.second_layer_nodes > 0:
            self._model.add(Dense(self._config.second_layer_nodes,
                                  activation=self._config.activation,
                                  kernel_initializer='he_uniform'))
        if self._config.third_layer_nodes > 0:
            self._model.add(Dense(self._config.third_layer_nodes,
                                  activation=self._config.activation,
                                  kernel_initializer='he_uniform'))
        self._model.add(BatchNormalization())
        self._model.add(Dense(len(self._ACTION_SPACE),
                              activation="linear"))

        self._memory = SequentialMemory(limit=self._config.memory_limit,
                                        window_length=1)

        self._policy = None
        if self._config.policy == 'MaxBoltzmannQPolicy':
            self._policy = MaxBoltzmannQPolicy() # https://github.com/keras-rl/keras-rl/blob/master/rl/policy.py#L242
        elif self._config.policy == 'EpsGreedyQPolicy':
            self._policy = EpsGreedyQPolicy(eps=.1)
        elif self._config.policy == 'LinearAnnealedPolicy':
            self._policy = LinearAnnealedPolicy(
                EpsGreedyQPolicy(),
                attr="eps",
                value_max=.5,
                value_min=0.025,
                value_test=0,
                nb_steps=self._config.NB_TRAINING_STEPS,
            )

        # Defining our DQN
        self._dqn = DQNAgent(
            model=self._model,
            nb_actions=self.action_space_size(),
            policy=self._policy,
            memory=self._memory,
            nb_steps_warmup=self._config.warmup_steps,
            gamma=self._config.gamma, # This is the discount factor for the Value we learn - we care a lot about future rewards, and we dont rush to get there
            target_model_update=self._config.target_model_update, # This controls how much/when our model updates: https://github.com/keras-rl/keras-rl/issues/55
            delta_clip=self._config.delta_clip, # Helps define Huber loss - cips values to be -1 < x < 1. https://srome.github.io/A-Tour-Of-Gotchas-When-Implementing-Deep-Q-Networks-With-Keras-And-OpenAi-Gym/
            enable_double_dqn=True,
        )

        self._dqn.compile(
            tf.keras.optimizers.Adam(lr=self._config.learning_rate),
            metrics=[
                tf.keras.metrics.MeanSquaredError(),
                tf.keras.metrics.MeanAbsoluteError(),
            ]
        )
        """

        # # Get initializer for hidden layers
        # init = tf.keras.initializers.RandomNormal(mean=.1, stddev=.02)
        #
        # # Input Layer; this shape is one that just works
        # self._model.add(Dense(512, input_shape=self.embedding_space, activation="relu",
        #                       use_bias=False, kernel_initializer='he_uniform', name='first_hidden'))
        #
        # # Hidden Layers
        # self._model.add(Flatten(name='flatten'))  # Flattening resolve potential issues that would arise otherwise
        # self._model.add(Dense(256, activation="relu", use_bias=False, kernel_initializer='he_uniform', name='second_hidden'))
        #
        # # Output Layer
        # self._model.add(Dense(len(self._ACTION_SPACE), use_bias=False, kernel_initializer=init, name='final'))
        # self._model.add(
        #     BatchNormalization())  # Increases speed: https://www.dlology.com/blog/one-simple-trick-to-train-keras-model-faster-with-batch-normalization/
        # self._model.add(Activation(
        #     "linear"))  # Same as passing activation in Dense Layer, but allows us to access last layer: https://stackoverflow.com/questions/40866124/difference-between-dense-and-activation-layer-in-keras

        # # This is how many battles we'll remember before we start forgetting old ones
        # self._memory = SequentialMemory(limit=max(self.num_battles, 10000), window_length=1)
        #
        # # Simple epsilon greedy policy
        # # This takes the output of our NeuralNet and converts it to a value
        # # Softmax is another probabilistic option: https://github.com/keras-rl/keras-rl/blob/master/rl/policy.py#L120
        # self._policy = LinearAnnealedPolicy(
        #     EpsGreedyQPolicy(eps=.1),
        #     attr="eps",
        #     value_max=1.0,
        #     value_min=0.05,
        #     value_test=0,
        #     nb_steps=self.num_battles,
        # )
        #
        # # Defining our DQN
        # self._dqn = DQNAgent(
        #     model=self._model,
        #     nb_actions=len(self._ACTION_SPACE),
        #     policy=self._policy,
        #     memory=self._memory,
        #     nb_steps_warmup=max(1000, int(self.num_battles / 10)),
        #     # The number of battles we go through before we start training: https://hub.packtpub.com/build-reinforcement-learning-agent-in-keras-tutorial/
        #     gamma=0.8,  # This is the discount factor for the Value we learn - we care a lot about future rewards
        #     target_model_update=.01,
        #     # This controls how much/when our model updates: https://github.com/keras-rl/keras-rl/issues/55
        #     delta_clip=1,
        #     # Helps define Huber loss - cips values to be -1 < x < 1. https://srome.github.io/A-Tour-Of-Gotchas-When-Implementing-Deep-Q-Networks-With-Keras-And-OpenAi-Gym/
        #     enable_double_dqn=True,
        # )
        #
        # self._dqn.compile(
        #     tf.keras.optimizers.Adam(lr=.001),
        #     metrics=[
        #         tf.keras.metrics.MeanSquaredError(),
        #         tf.keras.metrics.MeanAbsoluteError(),
        #     ]
        # )

    @property
    def model(self) -> training.Model:
        """
        Return our Keras-trained model
        """
        return self._model

    @property
    def memory(self) -> Memory:
        """
        Return the memory for our DQN
        """
        return self._memory

    @property
    def policy(self) -> Policy:
        """
        Return our policy for our DQN
        """
        return self._policy

    @property
    def dqn(self) -> DQNAgent:
        """
        Return our DQN object
        """
        return self._dqn

    def calc_reward(self, last_battle, current_battle) -> float:
        return self.reward_computing_helper(
            current_battle, fainted_value=2.0, hp_value=1.0, victory_value=30.0
        )

    def embed_battle(self, battle: AbstractBattle) -> ObservationType:
        return self._embedder.embed_battle(battle)

    def describe_embedding(self) -> Space:
        return self._embedder.describe_embedding()

    def set_opponent(self, opponent: Union[Player, str, List[Player], List[str]]):
        """
        Sets the next opponent to the specified opponent.

        :param opponent: The next opponent to challenge
        :type opponent: Player or str or a list of these
        """
        if isinstance(opponent, list):
            for i_opponent in opponent:
                if not isinstance(i_opponent, Player) and not isinstance(i_opponent, str):
                    raise RuntimeError(f"Expected type Player or str. Got {type(opponent)}")
        else:
            if not isinstance(opponent, Player) and not isinstance(opponent, str):
                raise RuntimeError(f"Expected type Player or str. Got {type(opponent)}")
        with self.opponent_lock:
            self.opponent = opponent

    # TODO: Test this
    def train(self, opponent: Union[Player, str, List[Player], List[str]], num_steps: int) -> None:

        self.set_opponent(opponent)
        self.start_challenging()
        self._dqn.fit(env=self, nb_steps=num_steps, callbacks=[WandbCallback()])


    # TODO: test this
    def evaluate_model(self, num_battles: int, visualize=False, verbose=False, verbose_end=False) -> float:
        self.reset_battles()
        self._dqn.test(env=self, nb_episodes=num_battles, visualize=visualize, verbose=verbose)
        if verbose_end: print("DQN Evaluation: %d wins out of %d battles" % (self.n_won_battles, num_battles))
        return self.n_won_battles * 1. / num_battles


    def save_model(self, filename=None) -> None:
        if filename is not None:
            self._dqn.save_weights("models/" + filename, overwrite=True)
        else:
            self._dqn.save_weights("models/model_" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S"), overwrite=True)

    def load_model(self, filename: str) -> None:
        self._dqn.load_weights("models/" + filename)
