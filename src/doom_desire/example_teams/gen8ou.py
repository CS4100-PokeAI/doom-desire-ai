import numpy as np

from poke_env.teambuilder.teambuilder import Teambuilder


class RandomTeamFromPool(Teambuilder):
    def __init__(self, teams):
        self.teams = [self.join_team(self.parse_showdown_team(team)) for team in teams]

    def yield_team(self):
        return np.random.choice(self.teams)

team_1 = """
Poseidon-Blade (Urshifu-Rapid-Strike) (M) @ Choice Band  
Ability: Unseen Fist  
EVs: 4 HP / 252 Atk / 252 Spe  
Jolly Nature  
- Surging Strikes  
- U-turn  
- Close Combat  
- Aqua Jet  

Temple (Ferrothorn) (M) @ Leftovers  
Ability: Iron Barbs  
Shiny: Yes  
EVs: 248 HP / 8 Def / 252 SpD  
Careful Nature  
- Spikes  
- Leech Seed  
- Knock Off  
- Iron Head  

Globox (Seismitoad) (M) @ Life Orb  
Ability: Swift Swim  
Shiny: Yes  
EVs: 252 SpA / 4 SpD / 252 Spe  
Modest Nature  
IVs: 0 Atk  
- Weather Ball  
- Stealth Rock  
- Earth Power  
- Focus Blast  

Thor (Thundurus-Therian) @ Heavy-Duty Boots  
Ability: Volt Absorb  
Shiny: Yes  
EVs: 252 SpA / 4 SpD / 252 Spe  
Timid Nature  
IVs: 0 Atk  
- Thunder  
- Focus Blast  
- Weather Ball  
- Nasty Plot  

Messenger (Pelipper) (M) @ Damp Rock  
Ability: Drizzle  
Shiny: Yes  
EVs: 248 HP / 252 Def / 8 SpD  
Bold Nature  
- Scald  
- U-turn  
- Defog  
- Roost  

poppinpepe 0_o (Ludicolo) @ Life Orb  
Ability: Swift Swim  
EVs: 252 Atk / 4 SpD / 252 Spe  
Adamant Nature  
- Swords Dance  
- Waterfall  
- Seed Bomb  
- Drain Punch  
"""

team_2 = """
Temple (Ferrothorn) (M) @ Leftovers  
Ability: Iron Barbs  
Shiny: Yes  
EVs: 248 HP / 252 Def / 4 SpD / 4 Spe  
Bold Nature  
IVs: 0 Atk  
- Substitute  
- Leech Seed  
- Iron Defense  
- Body Press  

Prophet (Necrozma) @ Power Herb  
Ability: Prism Armor  
Shiny: Yes  
EVs: 252 SpA / 4 SpD / 252 Spe  
Modest Nature  
- Meteor Beam  
- Photon Geyser  
- Psyshock  
- Heat Wave  

Moltres-Galar @ Heavy-Duty Boots  
Ability: Berserk  
EVs: 192 HP / 252 SpA / 64 Spe  
Modest Nature  
IVs: 0 Atk  
- Nasty Plot  
- Agility  
- Hurricane  
- Fiery Wrath  

Garchomp @ Leftovers  
Ability: Rough Skin  
EVs: 252 Atk / 4 SpD / 252 Spe  
Jolly Nature  
- Swords Dance  
- Stone Edge  
- Earthquake  
- Stealth Rock  

Raven (Tapu Fini) @ Choice Scarf  
Ability: Misty Surge  
Shiny: Yes  
EVs: 252 SpA / 4 SpD / 252 Spe  
Timid Nature  
IVs: 0 Atk  
- Trick  
- Defog  
- Moonblast  
- Surf  

Blaziken @ Air Balloon  
Ability: Speed Boost  
EVs: 252 Atk / 4 SpD / 252 Spe  
Jolly Nature  
- Swords Dance  
- Close Combat  
- Flare Blitz  
- Earthquake  
"""

# custom_builder = RandomTeamFromPool([team_1, team_2])
