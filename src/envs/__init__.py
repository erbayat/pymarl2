from functools import partial
import sys
import os

from .multiagentenv import MultiAgentEnv

from .starcraft import StarCraft2Env
from .matrix_game import OneStepMatrixGame
from .stag_hunt import StagHunt
from .uav_env import UAVEnv
from .partial_uav_env import PartialUAVEnv
from .test_env import TestEnv

try:
    gfootball = True
    from .gfootball import GoogleFootballEnv
except Exception as e:
    gfootball = False
    print(e)

def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)

REGISTRY = {}
REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env)
REGISTRY["stag_hunt"] = partial(env_fn, env=StagHunt)
REGISTRY["one_step_matrix_game"] = partial(env_fn, env=OneStepMatrixGame)
REGISTRY["uav_env"] = partial(env_fn, env=UAVEnv)
REGISTRY["partial_uav_env"] = partial(env_fn, env=PartialUAVEnv)
REGISTRY["test_env"] = partial(env_fn, env=TestEnv)


if gfootball:
    REGISTRY["gfootball"] = partial(env_fn, env=GoogleFootballEnv)


