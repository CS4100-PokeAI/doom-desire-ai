from datetime import datetime
from typing import Optional, Union, List

import wandb as wandb
from gym import Space
from tensorflow.python import keras
from wandb.keras import WandbCallback

from doom_desire.embed.abstract_embedder import AbstractEmbedder
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

from tensorflow.keras.layers import Dense, Flatten
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

    # async def env_move(self, battle: AbstractBattle):
    #
    #     if not self.current_battle or self.current_battle.finished:
    #         self.current_battle = battle
    #     if not self.current_battle == battle:  # pragma: no cover
    #         raise RuntimeError("Using different battles for queues")
    #     battle_to_send = self.user_funcs.embed_battle(battle)
    #     await self.observations.async_put(battle_to_send)
    #     action = await self.actions.async_get()
    #     if action == -1:
    #         return ForfeitBattleOrder()
    #     return self.user_funcs.action_to_move(action, battle)


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

        self._model.summary()

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

    def visualize_model(self):  # TODO: Does not work due to some issue with Graphviz and pydot
        keras.utils.plot_model(self._model, "model_visualization.png", show_shapes=True)

