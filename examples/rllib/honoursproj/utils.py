#----------------------------------------------------------------------------
# Author  : Sashen Moodley
# Student Number: 219006946
# ---------------------------------------------------------------------------
"""
Various utility functions for the project.

- boolean_string: Utility function for parsing boolean arguments from the command line.
- load_learned_model: Utility function for loading a trained model and running it a rendered environment.
"""

import copy
import argparse
import logging

import numpy as np

import dm_env
from dmlab2d.ui_renderer import pygame

import ray
from ray.rllib.agents.registry import get_trainer_class
from ray.tune.analysis.experiment_analysis import ExperimentAnalysis
from ray.rllib.policy.policy import PolicySpec
from ray.tune import tune
from ray.tune.registry import register_env

from examples.rllib import utils
from meltingpot.python import substrate

def boolean_string(s):
    """Utility function for parsing boolean arguments from the command line."""
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

def load_learned_model(substrate: str, algorithm: str, use_policy_sharing: True, experiment_path:str):
    """Utility function for loading a trained model and running it a rendered environment."""
    
    if substrate != 'clean_up':
        register_env('prisoners_dilemma', utils.env_creator)
    else:
        register_env('clean_up', utils.env_creator)
    
    # Loading the experiment
    experiment = ExperimentAnalysis(
        experiment_path,
        default_metric="episode_reward_mean",
        default_mode="max")

    config = experiment.best_config
    checkpoint_path = experiment.best_checkpoint

    trainer = get_trainer_class(algorithm)(config=config)
    trainer.restore(checkpoint_path)

    # Create a new environment to visualise
    env = utils.env_creator(config["env_config"]).get_dmlab2d_env()

    # Assigning policies to agents
    num_bots = config["env_config"]["num_players"]
    if use_policy_sharing:
        bots = [utils.RayModelPolicy(trainer, "av")] * num_bots
    else:
        bots = [utils.RayModelPolicy(trainer, f"player_{i}") for i in range(num_bots)]

    timestep = env.reset()
    states = [bot.initial_state() for bot in bots]
    actions = [0] * len(bots)

    # Configuring the PyGame renderer
    scale = 4
    fps = 5

    pygame.init()
    clock = pygame.time.Clock()
    pygame.display.set_caption("DM Lab2d")
    obs_spec = env.observation_spec()
    shape = obs_spec[0]["WORLD.RGB"].shape
    game_display = pygame.display.set_mode(
        (int(shape[1] * scale), int(shape[0] * scale)))

    # Running the environment
    for _ in range(config["horizon"]):
        obs = timestep.observation[0]["WORLD.RGB"]
        obs = np.transpose(obs, (1, 0, 2))
        surface = pygame.surfarray.make_surface(obs)
        rect = surface.get_rect()
        surf = pygame.transform.scale(surface,
                                    (int(rect[2] * scale), int(rect[3] * scale)))

        game_display.blit(surf, dest=(0, 0))
        pygame.display.update()
        clock.tick(fps)

        for i, bot in enumerate(bots):
            timestep_bot = dm_env.TimeStep(
                step_type=timestep.step_type,
                reward=timestep.reward[i],
                discount=timestep.discount,
                observation=timestep.observation[i])

        actions[i], states[i] = bot.step(timestep_bot, states[i])

        timestep = env.step(actions)