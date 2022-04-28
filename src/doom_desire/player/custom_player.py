from datetime import datetime
from typing import Optional, Union, List

from gym import Space
from tensorflow.python import keras
from wandb.keras import WandbCallback

from doom_desire.helpers.reward_calculator import RewardCalculator
from doom_desire.models.battle_modeler import AbstractBattleModeler
from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.player.env_player import Gen8EnvSinglePlayer
from poke_env.player.openai_api import ObservationType
from poke_env.player.player import Player
from poke_env.player_configuration import PlayerConfiguration
from poke_env.server_configuration import ServerConfiguration
from poke_env.teambuilder.teambuilder import Teambuilder

from rl.agents.dqn import AbstractDQNAgent
from rl.policy import Policy
from tensorflow.python.keras.engine import training
from rl.memory import Memory


class CustomRLPlayer(Gen8EnvSinglePlayer):

    def __init__(
            self,
            battle_format: Optional[str] = None,
            battle_modeler: AbstractBattleModeler = None,
            reward_calculator: RewardCalculator = None,
            *,
            build_model: Optional[bool] = True,
            use_wandb: Optional[bool] = False,

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
        self._use_wandb = use_wandb
        self._battle_modeler = battle_modeler
        self._reward_calculator = reward_calculator

        if build_model:
            self._create_model()

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

    def _create_model(self):

        self._model, self._policy, self._memory, self._dqn = self._battle_modeler.build_model(self.action_space_size())

    def calc_reward(self,
                    last_battle: AbstractBattle,
                    current_battle: AbstractBattle
                    ) -> float:

        if current_battle not in self._reward_buffer:
            self._reward_buffer[current_battle] = self._reward_calculator.starting_value

        current_value = self._reward_calculator.calc_reward(battle=current_battle)

        to_return = current_value - self._reward_buffer[current_battle]
        self._reward_buffer[current_battle] = current_value

        return to_return

    def embed_battle(self, battle: AbstractBattle) -> ObservationType:
        return self._battle_modeler.model_embedding(battle)  # This does not need to be singular, can be a list/tuple

    def describe_embedding(self) -> Space:
        return self._battle_modeler.describe_embedding()

    def set_opponent(self, opponent: Union[Player, str, List[Player], List[str]]):
        """
        Sets the next opponent to the specified opponent or the list of opponents (who will be randomly selected from)

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

    def train(self, opponent: Union[Player, str, List[Player], List[str]],
              num_steps: int, in_order: Optional[bool]=False) -> None:

        self._model.summary()

        if in_order:
            for single_opponent in opponent:
                self._train_helper(single_opponent, num_steps)
        else:
            self._train_helper(opponent, num_steps)

    def _train_helper(
            self,
            opponent: Union[Player, str, List[Player], List[str]],
            num_steps: int):

        self.set_opponent(opponent)
        if not self.challenge_task:  # haven't already started challenging opponents
            self.start_challenging()

        if not self._use_wandb:
            self._dqn.fit(env=self, nb_steps=num_steps)
        else:
            self._dqn.fit(env=self, nb_steps=num_steps, callbacks=[WandbCallback(monitor="val_loss",
                                                                                 verbose=0,
                                                                                 mode="auto",
                                                                                 save_weights_only=False,
                                                                                 log_weights=True,  # changed
                                                                                 log_gradients=False,
                                                                                 save_model=True,
                                                                                 training_data=None,
                                                                                 validation_data=None,
                                                                                 labels=[],
                                                                                 predictions=36,
                                                                                 generator=None,
                                                                                 input_type=None,
                                                                                 output_type=None,
                                                                                 log_evaluation=False,
                                                                                 validation_steps=None,
                                                                                 class_colors=None,
                                                                                 log_batch_frequency=None,
                                                                                 log_best_prefix="best_",
                                                                                 save_graph=False,
                                                                                 validation_indexes=None,
                                                                                 validation_row_processor=None,
                                                                                 prediction_row_processor=None,
                                                                                 infer_missing_processors=True,
                                                                                 log_evaluation_frequency=0,)])

    def evaluate_model(self, opponent: Player, num_battles: int, visualize=False, verbose=False, verbose_end=False) -> float:

        if not self.challenge_task:  # haven't already started challenging opponents
            self.start_challenging()
        self.reset_env(restart=True, opponent=opponent)

        self._dqn.test(env=self, nb_episodes=num_battles, verbose=verbose, visualize=visualize)
        if verbose_end:
            print(
            f"DQN Evaluation: {self.n_won_battles} victories out of {self.n_finished_battles} battles\n"
            )
        return self.n_won_battles * 1. / num_battles

    def test(self, num_battles: int, visualize=False, verbose=False):
        self._dqn.test(env=self, nb_episodes=num_battles, verbose=verbose, visualize=visualize)


    def save_model(self, filename=None) -> None:
        if filename is not None:
            self._dqn.save_weights("models/" + filename, overwrite=True)
        else:
            self._dqn.save_weights("models/model_" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + ".hdf5",
                                   overwrite=True)

    def load_model(self, filename: str) -> None:
        self._dqn.load_weights("models/" + filename)

    def visualize_model(self):  # TODO: Does not work due to some issue with Graphviz and pydot
        keras.utils.plot_model(self._model, "model_visualization.png", show_shapes=True)

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
    def dqn(self) -> AbstractDQNAgent:
        """
        Return our DQN object
        """
        return self._dqn
