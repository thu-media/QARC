import logging
from gym.envs.registration import registry, register, make, spec
logger = logging.getLogger(__name__)
# Algorithmic
# ----------------------------------------

register(
    id='QARC-v0',
    entry_point='gym_qarc.envs:QARCEnv',
)