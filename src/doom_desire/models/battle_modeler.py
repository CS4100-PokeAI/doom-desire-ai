from abc import ABC, abstractmethod
from typing import Tuple, TypeVar, Union

import wandb
from gym import Space
from rl.agents.dqn import AbstractDQNAgent, DQNAgent
from rl.memory import Memory, SequentialMemory
from rl.policy import Policy, MaxBoltzmannQPolicy, EpsGreedyQPolicy, LinearAnnealedPolicy
from tensorflow.python.keras.engine import training

from tensorflow.keras.models import Sequential
from tensorflow.python.keras.layers import Input, Dense, Flatten
from tensorflow.python.keras.optimizers import Adam

from doom_desire.embed.abstract_embedder import AbstractEmbedder
from poke_env.environment.abstract_battle import AbstractBattle

ObservationType = TypeVar("ObservationType")

class AbstractBattleModeler(ABC):

    @abstractmethod
    def build_model(self, action_space_size: int) -> Tuple[training.Model, Policy, Memory, AbstractDQNAgent]:
        pass

    @abstractmethod
    def model_embedding(self, battle: AbstractBattle) -> ObservationType:
        pass

    @abstractmethod
    def describe_embedding(self) -> Union[Space, Tuple[Space]]:
        pass



class SequentialBattleModeler(AbstractBattleModeler):

    def __init__(self,
                 config: wandb.wandb_sdk.Config,
                 embedder: AbstractEmbedder,
                 ):
        self._config = config
        self._embedder = embedder

        self._model = None
        self._policy = None
        self._memory = None
        self._dqn = None


    def _build_model(self, action_space_size: int) -> Tuple[Sequential, Policy, SequentialMemory, DQNAgent]:

        self._model = Sequential()
        self._model.add(Dense(self._config.first_layer_nodes,  # 128
                              activation=self._config.activation,  # 'elu'
                              input_shape=self._embedder.embedding_shape()))

        # Flattening resolve potential issues that would arise otherwise
        self._model.add(Flatten())
        if self._config.second_layer_nodes > 0:
            self._model.add(Dense(self._config.second_layer_nodes,  # 64
                                  activation=self._config.activation))  # 'elu'

        if self._config.third_layer_nodes > 0:
            self._model.add(Dense(self._config.third_layer_nodes,
                                  activation=self._config.activation,
                                  kernel_initializer='he_uniform'))

        self._model.add(Dense(units=action_space_size, activation="linear"))

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
            nb_actions=action_space_size,
            policy=self._policy,
            memory=self._memory,
            nb_steps_warmup=self._config.warmup_steps,  # 1000
            gamma=self._config.gamma,  # 0.5
            target_model_update=self._config.target_model_update,  # 1
            delta_clip=self._config.delta_clip,  # 0.01
            enable_double_dqn=True,
        )
        self._dqn.compile(Adam(clipvalue=1.0, lr=self._config.learning_rate),  # learning_rate=0.00025
                          metrics=['mean_squared_error',
                                   'mean_absolute_error',
                                   'mean_absolute_percentage_error',
                                   'cosine_proximity'])

        return self._model, self._policy, self._memory, self._dqn

    def build_model(self, action_space_size: int) -> Tuple[Sequential, Policy, SequentialMemory, DQNAgent]:

        if all([self._model, self._policy, self._memory, self._dqn]):
            return self._model, self._policy, self._memory, self._dqn
        else:
            return self._build_model(action_space_size)

    def model_embedding(self, battle: AbstractBattle) -> ObservationType:
        return self._embedder.embed_battle(battle)

    def describe_embedding(self) -> Space:
        return self._embedder.describe_embedding()
