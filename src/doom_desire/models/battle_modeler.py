from abc import ABC, abstractmethod
from types import SimpleNamespace
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
                 config: SimpleNamespace,
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


        # Simple model where only one layer feeds into the next

        # """
        # self._model = Sequential()
        # self._model.add(Dense(self._config.first_layer_nodes,
        #                       activation=self._config.activation,
        #                       input_shape=self._embedder.embedding_shape(),
        #                       kernel_initializer='he_uniform'))
        # self._model.add(Flatten())
        # if self._config.second_layer_nodes > 0:
        #     self._model.add(Dense(self._config.second_layer_nodes,
        #                           activation=self._config.activation,
        #                           kernel_initializer='he_uniform'))
        # if self._config.third_layer_nodes > 0:
        #     self._model.add(Dense(self._config.third_layer_nodes,
        #                           activation=self._config.activation,
        #                           kernel_initializer='he_uniform'))
        # self._model.add(BatchNormalization())
        # self._model.add(Dense(len(self._ACTION_SPACE),
        #                       activation="linear"))
        #
        # self._memory = SequentialMemory(limit=self._config.memory_limit,
        #                                 window_length=1)
        #
        # self._policy = None
        # if self._config.policy == 'MaxBoltzmannQPolicy':
        #     self._policy = MaxBoltzmannQPolicy() # https://github.com/keras-rl/keras-rl/blob/master/rl/policy.py#L242
        # elif self._config.policy == 'EpsGreedyQPolicy':
        #     self._policy = EpsGreedyQPolicy(eps=.1)
        # elif self._config.policy == 'LinearAnnealedPolicy':
        #     self._policy = LinearAnnealedPolicy(
        #         EpsGreedyQPolicy(),
        #         attr="eps",
        #         value_max=.5,
        #         value_min=0.025,
        #         value_test=0,
        #         nb_steps=self._config.NB_TRAINING_STEPS,
        #     )
        #
        # # Defining our DQN
        # self._dqn = DQNAgent(
        #     model=self._model,
        #     nb_actions=self.action_space_size(),
        #     policy=self._policy,
        #     memory=self._memory,
        #     nb_steps_warmup=self._config.warmup_steps,
        #     gamma=self._config.gamma, # This is the discount factor for the Value we learn - we care a lot about future rewards, and we dont rush to get there
        #     target_model_update=self._config.target_model_update, # This controls how much/when our model updates: https://github.com/keras-rl/keras-rl/issues/55
        #     delta_clip=self._config.delta_clip, # Helps define Huber loss - cips values to be -1 < x < 1. https://srome.github.io/A-Tour-Of-Gotchas-When-Implementing-Deep-Q-Networks-With-Keras-And-OpenAi-Gym/
        #     enable_double_dqn=True,
        # )
        #
        # self._dqn.compile(
        #     tf.keras.optimizers.Adam(lr=self._config.learning_rate),
        #     metrics=[
        #         tf.keras.metrics.MeanSquaredError(),
        #         tf.keras.metrics.MeanAbsoluteError(),
        #     ]
        # )
        # """

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
