# CS4100: PokeAI

# Onboarding

Getting Familiar with The Problem

- Definantly watch videos from [The Third Build](#the-third-build-youtube) and [Rempton Games](#rempton-games)
- Read some [articles and papers](#articles-and-pages) about other attempts in the same realm
- Check out some of the [repositories](#repositories) that we are using or are good references
- [github-how-to-make-a-fork-of-public-repository-private](https://stackoverflow.com/a/66406865) and [github-pull-request-from-private-to-public-repo-possible](https://stackoverflow.com/a/66414498)


# References


### Videos

---

### The Third Build YouTube
See more for [The Third Build](#the-third-build)

[YouTube Channel](https://www.youtube.com/channel/UCdwshbwxNBoCCBoZGgf3U6Q)

- [How an A.I. is Becoming the World's Best Pokemon Player](https://www.youtube.com/watch?v=rhvj7CmTRkg&t=1129s&ab_channel=TheThirdBuild)
- [I said some things wrong about my A.I.... Let's Fix That!](https://www.youtube.com/watch?v=RbBJ_J89wso&t=73s&ab_channel=TheThirdBuild)
- [Why These Pokemon are an A.I.'s Biggest Threat](https://www.youtube.com/watch?v=vjQi8V96_FI)

---

### Rempton Games

- [Programming AI for Pokemon Showdown + Bot Battle Royale!](https://youtu.be/C1KpQc9cWmM)

and its [article](https://remptongames.com/2021/06/27/programming-ai-for-pokemon-showdown-bot-battle-royale/), which used to be on [Game Developer](https://www.gamedeveloper.com/disciplines/programming-ai-for-pokemon-showdown-bot-battle-royale-) but link broke some time in March 2022

---

## Articles and Pages

- [The Third Build](#the-third-build)
- [poke-env docs](#docs)
- vasumv/pokemon_ai on [Smogon](https://www.smogon.com/forums/threads/pokemon-showdown-ai-bot.3547689/) [Stunfisk Reddit](https://www.reddit.com/r/stunfisk/comments/3i4hww/pokemon_showdown_ai/) for [Github repo](#vasumv)
- [Poke-Agent: Pokemon Battling & Machine Learning](https://towardsdatascience.com/poke-agent-pokemon-battling-reinforcement-learning-27ef10885c5c), which uses Q-Learning
- [THE QUEST TO MAKE AN AI THAT CAN PLAY COMPETITIVE POKÉMON](https://www.theverge.com/2021/12/2/22811849/pokemon-competitive-ai-nintendo-game-freak)
- [Technical Machine](http://doublewise.net/pokemon/) an early attempt at a bot that did not do well (but impressive for 2010 using [Yomi layers](http://doublewise.net/pokemon/yomi.php))
- [Directory](https://coder.social/topic/pokemon-showdown-bot) of many different Pokemon Showdown bots that have been written 

### Academic Papers

- [Showdown AI competition](https://ieeexplore.ieee.org/document/8080435)
- [Metagrok](#metagrok) has very good paper and public repository
- [Competitive Deep Reinforcement Learning over a Pokémon Battling Simulator](https://ieeexplore.ieee.org/document/9096092)
- [VGC AI Competition - A New Model of Meta-Game Balance AI Competition](https://ieeexplore.ieee.org/document/9618985)
- [The 2016 Two-Player GVGAI Competition](https://ieeexplore.ieee.org/document/8100955) (not actually about pokemon)
- [Percymon: A Pokemon Showdown Artifical Intelligence](https://varunramesh.net/content/documents/cs221-final-report.pdf) for [Github repo](#rameshvarun-showdownbot)

---

### The Third Build

[Personal Website](https://aed3.github.io/) for creator of Pokemon Battle Predictor and Future Sight AI

[YouTube Channel](https://www.youtube.com/channel/UCdwshbwxNBoCCBoZGgf3U6Q) and [videos](#the-third-build-youtube)

[Pokemon Battle Predictor Website](https://www.pokemonbattlepredictor.com/) 
- [Battle Predictor](https://www.pokemonbattlepredictor.com/battle-predictor)
  - Chrome extension tool does not seem to be working
- [Future Sight AI](https://www.pokemonbattlepredictor.com/FSAI)
  - [HOW FUTURE SIGHT AI WORKS](https://www.pokemonbattlepredictor.com/FSAI/how-fsai-works)
  - [EXHIBITION STATS](https://www.pokemonbattlepredictor.com/FSAI/exhibition-stats)

Some public extras on [Github](https://github.com/aed3/FSAI-Extras) for Future Sight AI (most of his code is private)

- [How an A.I. is Becoming the World’s Best Pokemon Player](https://www.pokemoncrossroads.com/video-becoming-worlds-best-pokemon-player/)

---

## Repositories

### Using Directly

---

#### Showdown

Navigation: [Website][1] | [Server repository][2] | [Client repository][3] | [Dex repository][4]

  [1]: http://pokemonshowdown.com/
  [2]: https://github.com/smogon/pokemon-showdown
  [3]: https://github.com/smogon/pokemon-showdown-client
  [4]: https://github.com/Zarel/Pokemon-Showdown-Dex

Noteworthy Pages:

- [Custom Rules](https://github.com/smogon/pokemon-showdown/blob/master/config/CUSTOM-RULES.md)
- [Game Logging](https://github.com/smogon/pokemon-showdown/blob/master/logs/logging.md)
- [Simulator](https://github.com/smogon/pokemon-showdown/blob/master/sim/SIMULATOR.md)
- [Bot F.A.Q.](https://gist.github.com/Kaiepi/becc5d0ecd576f5e7733b57b4e3fa97e)

---

#### poke-env

[Github](https://github.com/hsahovic/poke-env)

Noteworthy repo links:

- [Project tracking](https://github.com/hsahovic/poke-env/projects) for [known issues](https://github.com/hsahovic/poke-env/projects/4) and [general](https://github.com/hsahovic/poke-env/projects/2)
- [rl_with_open_ai_gym_wrapper](https://github.com/hsahovic/poke-env/blob/master/examples/rl_with_open_ai_gym_wrapper.py)
- Certainly many more 


##### [Docs](https://poke-env.readthedocs.io/en/stable/index.html)

- [Getting Started](https://poke-env.readthedocs.io/en/stable/getting_started.html)
- [Creating Agents](https://poke-env.readthedocs.io/en/stable/getting_started.html#creating-agents)
- [Reinforcement learning with the OpenAI Gym wrapper](https://poke-env.readthedocs.io/en/stable/rl_with_open_ai_gym_wrapper.html#rl-with-open-ai-gym-wrapper)
- [Adapting the max player to gen 8 OU and managing team preview](https://poke-env.readthedocs.io/en/stable/ou_max_player.html)
- [Karas Docs](https://keras-rl.readthedocs.io/en/latest/agents/overview/)

---

### Other Attempts

---

#### pmariglia

Has repositories for [Showdown Bot](https://github.com/pmariglia/showdown) and [poke-env](https://github.com/pmariglia/showdown)

https://www.libhunt.com/r/pmariglia/showdown

These seem to be the closest things to what we are trying to do and definetly need to be looked into deeper

---

#### vasumv

vasumv/pokemon_ai seems very promising as well

- [Github](https://github.com/vasumv/pokemon_ai)

---

#### rameshvarun showdownbot

This one might be almost an exact copy, as it was also done for a class

- [Github](https://github.com/rameshvarun/showdownbot)

---

#### VGC AI Framework

This one seems incredibly good, is for VGC, uses Gym OpenAI, and is regularly maintained (update just days ago)

- [Gitlab](https://gitlab.com/DracoStriker/pokemon-vgc-engine)

---

#### CynthiAI

Similar seeming project but haven't looked into it

Looks like it doesn't work anymore but would have 

- [CynthiAI Github](https://github.com/Sisyphus25/CynthiAI)
- [CynthiAgent.js](https://github.com/Sisyphus25/CynthiAI/blob/master/CynthiAgent.js) might have some things we can use

---

#### taylorhansen pokemonshowdown-ai
- [Github](https://github.com/taylorhansen/pokemonshowdown-ai)

---

#### DeathlyPlays Pokemon-Showdown-Bot

written in JavaScript for Node

- [Github](https://github.com/DeathlyPlays/Pokemon-Showdown-Bot)

---

#### Metagrok

Very important for furutre work, has really good paper

- [Github](https://github.com/yuzeh/metagrok)
- [Paper](https://www.yuzeh.com/assets/CoG-2019-Pkmn.pdf)


----


## Pokemon Resources

[Smogon Usage Stats](https://www.smogon.com/stats/)

[Damage Calculator](https://calc.pokemonshowdown.com/) and its [Github repo](https://github.com/smogon/damage-calc)

[Random Battles Competitive Guide](https://www.smogon.com/articles/randbats-guide-gen8) might have some tricks we can implement

Places from which we can grab good sample teams for the bots to use

#### PokeAIM Teams [website](https://www.pokeaimmd.com/teams)



&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;


# The pokemon showdown Python environment

[![PyPI version fury.io](https://badge.fury.io/py/poke-env.svg)](https://pypi.python.org/pypi/poke-env/)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/poke-env.svg)](https://pypi.python.org/pypi/poke-env/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation Status](https://readthedocs.org/projects/poke-env/badge/?version=stable)](https://poke-env.readthedocs.io/en/stable/?badge=stable)

A Python interface to create battling pokemon agents. `poke-env` offers an easy-to-use interface for creating rule-based or training Reinforcement Learning bots to battle on [pokemon showdown](https://pokemonshowdown.com/).

![A simple agent in action](rl-gif.gif)

# Getting started

Agents are instance of python classes inheriting from `Player`. Here is what your first agent could look like:

```python
class YourFirstAgent(Player):
    def choose_move(self, battle):
        for move in battle.available_moves:
            if move.base_power > 90:
                # A powerful move! Let's use it
                return self.create_order(move)

        # No available move? Let's switch then!
        for switch in battle.available_switches:
            if switch.current_hp_fraction > battle.active_pokemon.current_hp_fraction:
                # This other pokemon has more HP left... Let's switch it in?
                return self.create_order(switch)

        # Not sure what to do?
        return self.choose_random_move(battle)
```

To get started, take a look at [our documentation](https://poke-env.readthedocs.io/en/stable/)!


## Documentation and examples

Documentation, detailed examples and starting code can be found [on readthedocs](https://poke-env.readthedocs.io/en/stable/).


## Installation

This project requires python >= 3.6 and a [Pokemon Showdown](https://github.com/Zarel/Pokemon-Showdown) server.

```
pip install poke-env
```

You can use [smogon's server](https://play.pokemonshowdown.com/) to try out your agents against humans, but having a development server is strongly recommended. In particular, it is recommended to use the `--no-security` flag to run a local server with most rate limiting and throttling turned off. Please refer to [the docs](https://poke-env.readthedocs.io/en/stable/getting_started.html#configuring-a-showdown-server) for detailed setup instructions.


```
git clone https://github.com/smogon/pokemon-showdown.git
cd pokemon-showdown
npm install
cp config/config-example.js config/config.js
node pokemon-showdown start --no-security
```

### Development version

You can also clone the latest master version with:

```
git clone https://github.com/hsahovic/poke-env.git
```

Dependencies and development dependencies can then be installed with:

```
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

## Acknowledgements

This project is a follow-up of a group project from an artifical intelligence class at [Ecole Polytechnique](https://www.polytechnique.edu/).

You can find the original repository [here](https://github.com/hsahovic/inf581-project). It is partially inspired by the [showdown-battle-bot project](https://github.com/Synedh/showdown-battle-bot). Of course, none of these would have been possible without [Pokemon Showdown](https://github.com/Zarel/Pokemon-Showdown).

Team data comes from [Smogon forums' RMT section](https://www.smogon.com/).

## Data

Data files are adapted version of the `js` data files of [Pokemon Showdown](https://github.com/Zarel/Pokemon-Showdown).

## License
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Other

[![CircleCI](https://circleci.com/gh/hsahovic/poke-env.svg?style=svg)](https://circleci.com/gh/hsahovic/poke-env)
[![codecov](https://codecov.io/gh/hsahovic/poke-env/branch/master/graph/badge.svg)](https://codecov.io/gh/hsahovic/poke-env)
<a href="https://github.com/ambv/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
