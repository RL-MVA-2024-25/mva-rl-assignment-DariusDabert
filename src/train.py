from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient

import time
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
import json

import torch 
from imitation.algorithms import bc
from imitation.data import serialize
from imitation.data.rollout import (
    TrajectoryAccumulator,
    types,
    GenTrajTerminationFn,
    make_sample_until,
    spaces,
    rollout_stats,
    unwrap_traj,
    dataclasses,
)
from imitation.util.logger import configure as configure_logger
from imitation.util.util import save_policy
from imitation.data import rollout
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
import numpy as np
from gymnasium.wrappers import TimeLimit, TransformObservation
from coolname import generate_slug
from tqdm.rich import trange, tqdm

try:
    import wandb
    DISABLE_WANDB = False
except ImportError:
    DISABLE_WANDB = True


env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.


# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!

SAVE_PATH = Path(__file__).parent.parent/"src/models"
MODEL_NAME = "bc/1736775529_flying-tiger/repeat_0_best.pkl"


class ProjectAgent:
    def act(self, observation, use_random=False):
        observation = np.log(np.maximum(observation, 1e-8))
        return self.policy.predict(observation, deterministic=True)[0]

    def save(self, path):
        torch.save(self.policy, path)

    def load(self):
        self.policy = torch.load(SAVE_PATH / MODEL_NAME, weights_only=False)
