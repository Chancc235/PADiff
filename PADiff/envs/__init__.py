from .lbf.foraging import ForagingEnv
from .multiagentenv import MultiAgentEnv
from .overcooked.overcookedenv import OvercookedMultiEnv
from .starcraft2.starcraft2 import StarCraft2Env
from .stag_hunt import StagHunt

REGISTRY = {
    "sc2": StarCraft2Env,
    "overcooked": OvercookedMultiEnv,
    "lbf": ForagingEnv,
    "stag_hunt": StagHunt,
}
