from rl.agents import DQNAgent
from rl.memory import Memory
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from tensorflow.python.keras.optimizer_v2.adam import Adam


def example_dqn_structure(model, nb_actions: int, memory: Memory) -> DQNAgent:

    policy = LinearAnnealedPolicy(
        EpsGreedyQPolicy(),
        attr="eps",
        value_max=1.0,
        value_min=0.05,
        value_test=0.0,
        nb_steps=10000,
    )

    dqn = DQNAgent(
        model=model,
        nb_actions=nb_actions,
        policy=policy,
        memory=memory,
        nb_steps_warmup=1000,
        gamma=0.5,
        target_model_update=1,
        delta_clip=0.01,
        enable_double_dqn=True,
    )
    dqn.compile(Adam(learning_rate=0.00025), metrics=["mae"])

    return dqn



# This is the function that will be used to train the dqn
def example_dqn_training(player, dqn, nb_steps):
    dqn.fit(player, nb_steps=nb_steps)
    player.complete_current_battle()

