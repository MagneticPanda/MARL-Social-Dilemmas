#----------------------------------------------------------------------------
# Author  : Sashen Moodley
# Student Number: 219006946
# ---------------------------------------------------------------------------

""" Runs the training and evaluation of a substrate using a particular algorithm.

This script uses RLlib algorithms to train and evaluate a substrate. The algorithms
are configured to ensure a consistent training and evaluation process. Furthermore,
the parameters are favoured towards ensuring a stable training process. Adjust the
degree of parallelism (remote workers) according to your available resources.

This script is designed to be run from the command line. It takes in the following arguments:
    -- experiment_name: A custom name for the experiment.
    -- *substrate: The substrate to use for training and evaluation.
    -- *algorithm: The algorithm to use for training and evaluation.
    -- use_policy_sharing: Whether to use policy sharing or not (True / False).
    -- num_iterations: The number of iterations to train for.
    -- local_dir: The directory to store the experiment results in.
    -- use_gpu: Whether to use GPU training or not (True / False). 
    -- record_env: Whether to record the environment or not (True / False).

* Mandatory arguments.
Note: If you encounter a ModuleNotFoundError when running this script, ensure that you have
installed Melting Pot and that the PYTHONPATH environment variable is set correctly (look 
at the README for more information).

Example usage:
    python honoursproj/train_and_eval.py --experiment_name CustomExp --substrate clean_up --algorithm R2D2 --use_policy_sharing True --num_iterations 10 --local_dir testing/temp --use_gpu False --record_env True
"""

import copy
import argparse

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

import honoursproj.utils as honours_utils

class Experiment:
    """Experiment class to run the training and evaluation of a substrate."""
    
    # Can change this depending on your available resources
    REMOTE_WORKER_MAPPING = {
        "A3C": 2,
        "PPO": 1,
        "R2D2": 1,
        "A3C_NPS": 1,
        "PPO_NPS": 1,
        "R2D2_NPS": 0,
    }

    # Can change fragment lengths according to your preference
    ROLLOUT_FRAGMENT_LENGTH_MAPPING = {
        "A3C": 10,
        "PPO": 1,
        "R2D2": 4,
    }

    def __init__(self, experiment_name: str, substrate: str, algorithm: str, use_policy_sharing: bool,
        num_training_iter: int, local_dir: str, use_gpu: bool, record_env: bool):
        """Initializes the experiment.

        Args:
            - experiment_name: Name of the experiment.
            - substrate: Substrate to use for training.
            - algorithm: Algorithm to use for training.
            - use_policy_sharing: Whether to use policy sharing.
            - num_training_iter: Number of training iterations.
            - local_dir: Local directory to store trained policies and recordings.
            - use_gpu: Whether to use GPU.
            - record_env: Whether to record the environment.
            - logger: Logger to use.
        """
        self.experiment_name = experiment_name
        self.substrate = substrate
        self.algorithm = algorithm.upper()
        self.use_policy_sharing = use_policy_sharing
        self.num_training_iter = num_training_iter
        self.local_dir = local_dir
        self.use_gpu = use_gpu
        self.record_env = record_env

    @staticmethod
    def ann_model() -> dict:
        """Returns the ANN model configuration, used by all algorithms.
        Sizes are chosen to match the substrate specifications."""
        ann_config = {}
        ann_config["conv_filters"] = [[16, [8, 8], 8], [128, [11, 11], 1]]
        ann_config["conv_activation"] = "relu"
        ann_config["post_fcnet_hiddens"] = [256]
        ann_config["post_fcnet_activation"] = "relu"
        ann_config["use_lstm"] = True
        ann_config["lstm_use_prev_action"] = True
        ann_config["lstm_use_prev_reward"] = False
        ann_config["lstm_cell_size"] = 256
        return ann_config


    @property
    def experiment_config(self) -> dict:
        """Returns the full experiment configuration.
        For extensive details of each configuration parameter, refer to: 
        https://docs.ray.io/en/latest/rllib/rllib-training.html
        """

        # PART 1: Getting a base template for the config from the initial algorithm config
        config = copy.deepcopy(get_trainer_class(self.algorithm).get_default_config())

        # PART 2: Setting up the environment-specific configurations
        config["env_config"]=substrate.get_config(self.substrate)

        register_env(
            name=self.substrate if self.substrate =="clean_up" else "prisoners_dilemma", 
            env_creator=utils.env_creator)
        config["env"] = self.substrate if self.substrate =="clean_up" else "prisoners_dilemma"

        # PART 3: Configuration for multi-agent setup
        # Temp environment to extract the action and observation spaces
        test_env = utils.env_creator(env_config=config["env_config"])

        policies = {}
        policy_mapping_fn = None  # associates agents ids with with policy ids

        if self.use_policy_sharing: # Policy sharing uses the same policy for all agents
            policies["av"] = PolicySpec(
                    policy_class=None, # using the default policy
                    observation_space=test_env.observation_space["player_0"], 
                    action_space=test_env.action_space["player_0"], 
                    config={})
            policy_mapping_fn = lambda agent_id, episode, worker, **kwargs: "av"
        else:  # Each agents maintains and updates their own policy
            for player in range(len(test_env.observation_space)):
                policies["player_" + str(player)] = PolicySpec(
                    policy_class=None,
                    observation_space=test_env.observation_space["player_" + str(player)], 
                    action_space=test_env.action_space["player_" + str(player)], 
                    config={})
            policy_mapping_fn = lambda agent_id, episode, worker, **kwargs: str(agent_id)

        config["multiagent"] = {
            "policies": policies,
            "policy_mapping_fn": policy_mapping_fn,
            "count_steps_by": "env_steps"
        }

        # PART 4: Remote worker and resource configuration
        config["num_gpus"] = 1 if self.use_gpu else 0
        config["num_workers"] = self.REMOTE_WORKER_MAPPING[self.algorithm if self.use_policy_sharing 
            else self.algorithm + "_NPS"]
        
        # PART 5: Algorithm-specific configurations
        if self.algorithm == "PPO":
            config["sgd_minibatch_size"] = 128

        config["horizon"] = config["env_config"].lab2d_settings["maxEpisodeLengthFrames"]  # assigning episode length
        config["batch_mode"] = "complete_episodes"  # Ensures a reward is returned by enforcing that all episodes are completed        
        config["train_batch_size"] = config["env_config"].lab2d_settings["maxEpisodeLengthFrames"]  # concatenated experience size
        config["rollout_fragment_length"] = self.ROLLOUT_FRAGMENT_LENGTH_MAPPING[self.algorithm]  # size of each fragment of an episode
        config["no_done_at_end"] = False # Ensures that the last state of an episode is not treated as a terminal state

        # PART 6: ANN model configuration
        config["model"] = Experiment.ann_model()
        config["framework"] = "tf2" if self.algorithm == "R2D2" else "tf"

        # PART 7: Evaluation configuration
        config["evaluation_duration_unit"] = "episodes"
        config["evaluation_interval"] = 5  # Evaluate after every 5 training iterations.
        config["evaluation_duration"] = 5  # Evaluate 5 episodes per evaluation round.
        
        # PART 8: Miscellaneous configurations
        config["log_level"] = "DEBUG"
        config["record_env"] = self.record_env

        return config

        
    def run(self):
        """Runs the experiment."""

        # 1. GATHER THE CONFIGURATION
        print(f"[INFO] Setting up configuaration for: {self.experiment_name}...")
        config = self.experiment_config
        print("[INFO] Configuration set up successfully!")

        # 2. INITIALIZE RAY AND GPUS
        ray.init(num_gpus=config["num_gpus"])

        # 3. TRAINING WITH REAL-TIME OPTIMIZATION
        print("[INFO] Running trials and training agents...")
        tune.run (
            self.algorithm,
            config=config,
            stop= {'training_iteration': self.num_training_iter},
            metric="episode_reward_mean",  # metric to optimise for
            mode="max",  # trying to maximise the metric attribute
            local_dir=self.local_dir,
            checkpoint_freq=1,
            checkpoint_score_attr="episode_reward_mean",
            keep_checkpoints_num=5,
            checkpoint_at_end=True,
            name=self.experiment_name,
        )

        print(f"[INFO] Training complete for {self.experiment_name}!")


def main():
    # Setting up the arugment parser
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="CustomExperiment",
        help="The cutom name for the experiment.")
    parser.add_argument(
        "--substrate",
        type=str,
        default="clean_up",
        help="The substrate to use for training and evaluation."
        "Choose between 'clean_up' or 'prisoners_dilemma_in_the_matrix'.")
    parser.add_argument(
        "--algorithm",
        type=str,
        default="A3C",
        help="The algorithm to use for training and evaluation."
        "Choose between 'A3C', 'PPO', or 'R2D2'.")
    parser.add_argument(
        "--use_policy_sharing",
        type=honours_utils.boolean_string,
        default=True,
        help="Whether to use policy sharing or not ('True' / 'False').")
    parser.add_argument(
        "--num_iterations",
        type=int,
        default=100,
        help="The number of iterations to train for.")
    parser.add_argument(
        "--local_dir",
        type=str,
        default="testing/temp",
        help="The directory to store the experiment results in."
        "I have made an empty directory called 'testing/temp/' for you to use.")
    parser.add_argument(
        "--use_gpu",
        type=honours_utils.boolean_string,
        default=False,
        help="Whether to use GPU training or not. ('True' / 'False')"
        "Make sure you have CUDA installed and configured correctly.")
    parser.add_argument(
        "--record_env",
        type=honours_utils.boolean_string,
        default=False,
        help="Whether to record the environment or not. ('True' / 'False')"
        "Recording will be stored in same directory as the experiment.")

    args = parser.parse_args()

    # Initialising and running the experiment with the provided arguments
    experiment = Experiment(
        experiment_name=args.experiment_name,
        substrate=args.substrate,
        algorithm=args.algorithm,
        use_policy_sharing=args.use_policy_sharing,
        num_training_iter=args.num_iterations,
        local_dir=args.local_dir,
        use_gpu=args.use_gpu,
        record_env=args.record_env,
    )

    experiment.run()

if __name__ == "__main__":
    main()