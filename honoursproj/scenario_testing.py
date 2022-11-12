#----------------------------------------------------------------------------
# Author  : Sashen Moodley
# Student Number: 219006946
# ---------------------------------------------------------------------------

"""Tests the trained focal policies in a testing scenario and reports back on various custom social metrics.

This script evaluates the focal policies which were trained in the `train_and_eval.py` script.
The testig uses various Melting Pot scenarios to test the focal policies by introducing
previously unseen background agents. Thus, the learned policies governing the focal population
needs to be generic enough to handle the introduction of new agents.

To quantify the performance of the focal policies, I implemented a set of custom social metrics
defined in the `Social_Metrics` class. These metrics are based on the specification of [4] and 
[58] in my thesis rearch paper.

This script is designed to be run from the command line. It takes in the following arguments:
    --scenario_name: The name of the scenario to test. Must be in scenario.SCENARIOS_BY_SUBSTRATE.
        Clean Up scenarios take the form: 'clean_up_<0-6>'
        Prisoner's Dilemma scenarios take the form: 'prisoners_dilemma_in_the_matrix_<0-5>'
    --algorithm: The name of the algorithm to load and test.
    --use_policy_sharing: (True / False) Whether to use policy sharing.
        It is recommended to mirror the setting used during training.
    --substrate: The name of the base substrate to test (clean_up / prisoners_dilemma_in_the_matrix)
    --experiment_path: The path to the experiment directory where you saved the trained model.

Note: If you encounter a ModuleNotFoundError when running this script, ensure that you have
installed Melting Pot and that the PYTHONPATH environment variable is set correctly (look 
at the README for more information).

Example usage:
    python honoursproj/scenario_testing.py --scenario_name clean_up_2 --algorithm PPO --use_policy_sharing True --substrate clean_up --experiment_path testing/cu_ray_logs/CU_PPO
"""

import time
import argparse

from typing import List
import numpy as np

from meltingpot.python import scenario
import dm_env
from dmlab2d.ui_renderer import pygame

from ray.rllib.agents.registry import get_trainer_class
from ray.tune.analysis.experiment_analysis import ExperimentAnalysis
from ray.tune.registry import register_env

from examples.rllib import utils
from meltingpot.python.utils import bots

import honoursproj.utils as honours_utils


class Social_Metrics:
    """Class of the custom social metrics for algorithm generalization assessment.
    This includes:
        - Individual return
        - Per-capita return
        - Focal per-capita return
        - Background per-capita return
        - Utilitarian
    """
    
    def __init__(self, rewards) -> None:
        """
        Args:
            rewards: A dictionary of rewards for the focal and background population.
                The dictionary should have the following keys:
                    'Focal rewards': A list of lists of rewards for each focal agent.
                    'Background rewards': A list of lists of rewards for each background agent.
        """
        self.rewards = rewards


    def _merge_rewards(self) -> List[List[int]]:
        """Utility function to merge focal and background population rewards
        to single list. Indexed by scenario step(s).
        
        Returns:
            A list of episode rewards containing the concatenatated focal and
            background rewards for each episode.
        """

        merged_rewards = []
        for i in range(len(self.rewards['Focal rewards'])):
            merged_rewards.append(self.rewards['Focal rewards'][i] + self.rewards['Background rewards'][i])

        return merged_rewards


    def individual_return(self, player_id: int = None):
        """Performance for each individial policy. Total reward for player/agent i"""
        
        merged_rewards = self._merge_rewards()

        res = {}

        for i in range(len(merged_rewards[0])):
            res[f'player_{i}'] = 0
        
        for step_reward in merged_rewards:
            for i, player_reward in enumerate(step_reward):
                res[f'player_{i}'] += player_reward

        if player_id is not None:
            try:
                return res[f'player_{player_id}']
            except:
                raise IndexError(f"Tried to access player_{player_id}, but only {len(merged_rewards[0])} players")
        else:
            return res


    @property
    def per_capita_return(self):
        """Performance of the joint policy. Mean individual return across all N players"""
        return(np.average(np.array([reward for reward in self.individual_return().values()])))


    @property
    def focal_per_capita_return(self):
        """Performance of focal players. The mean return across all F focal players"""
        res = [0] * len(self.rewards['Focal rewards'][0])

        for focal_step_reward in self.rewards['Focal rewards']:
            for i, agent_reward in enumerate(focal_step_reward):
                res[i] += agent_reward
        
        return(np.average(np.array(res)))
    

    @property
    def background_per_capita_return(self):
        """Performance of the background players. The mean return across all B background players"""
        
        res = [0] * len(self.rewards['Background rewards'][0])

        for background_step_reward in self.rewards['Background rewards']:
            for i, bot_reward in enumerate(background_step_reward):
                res[i] += bot_reward
        
        return(np.average(np.array(res)))
    

    def utilitarian(self, steps_per_episode: int = 1000):
        """Also known as Efficiency, measures the sum total of all rewards obtained by all agents
        averaged over all timesteps. In CU and PD each episode consists of 1000 timesteps"""
        return np.sum(np.array([reward for reward in self.individual_return().values()]))/steps_per_episode


    @property
    def report(self):
        """Summary report of the social metrics for a testing scenario"""

        print("====================================================================================================================================")
        print("=================================================== Social Metrics =================================================================")
        print("====================================================================================================================================", end='\n\n')

        print("------------------------------------------------- Individual Return ----------------------------------------------------------------")
        print(self.individual_return())
        print("------------------------------------------------- Per Capita Return ----------------------------------------------------------------")
        print(self.per_capita_return)
        print("---------------------------------------------- Focal Per Capita Return -------------------------------------------------------------")
        print(self.focal_per_capita_return)
        print("------------------------------------------- Background Per Capita Return -----------------------------------------------------------")
        print(self.background_per_capita_return)
        print("---------------------------------------------------- Utilitarian -------------------------------------------------------------------")
        print(self.utilitarian(), end='\n\n')

        print("====================================================================================================================================")


class ScenarioTesting:
    """Tests the trained focal policies in a testing scenario"""
    def __init__(self, scenario_name: str, algorithm: str, use_policy_sharing: bool, substrate: str, experiment_path: str):
        """Initialize the ScenarioTesting class
        Args:
            - scenario_name (str): Name of the scenario to test (Must be in scenario.SCENARIOS_BY_SUBSTRATE)
                Clean Up scenarios take the form: 'clean_up_<0-6>'
                Prisoner's Dilemma scenarios take the form: 'prisoners_dilemma_in_the_matrix_<0-5>'
            - algorithm (str): Name of the algorithm to test
            - use_policy_sharing (bool): Whether to use policy sharing. It is recommended to mirror the setting
                that was used during training. Defaults to True for robustness.
            - substrate (str): Name of the substrate to test
            - experiment_path (str): Path to the experiment directory
        """

        self.scenario_name = scenario_name
        self.algorithm = algorithm.upper()
        self.use_policy_sharing = use_policy_sharing
        self.substrate = substrate
        self.experiment_path = experiment_path
        
        self.rewards = {
            'Focal rewards': [],
            'Background rewards': []
        }

        # Getting the scenario configuration, building the scenario, and registering the environment
        self.scenario_config = scenario.get_config(scenario_name=self.scenario_name)
        self.built_scenario = scenario.build(config=self.scenario_config)
        register_env(name=substrate if substrate == 'clean_up' else 'prisoners_dilemma', env_creator=utils.env_creator)


    @property
    def no_of_focal_players(self) -> int:
        return len(self.built_scenario.action_spec())


    @property
    def no_of_background_players(self) -> int:
        return self.built_scenario._background_population._population_size
    

    def load_experiment(self):
        """Load the experiment and extract policies for testing
        Returns:
            A list of policies for testing
        """

        # Loading the best-performing experiment from the provided experiment path
        experiment = ExperimentAnalysis(
            experiment_checkpoint_path=self.experiment_path,
            default_metric="episode_reward_mean",
            default_mode="max"
        )

        self.config = experiment.best_config
        checkpoint_path = experiment.best_checkpoint

        trainer = get_trainer_class(self.algorithm)(config=self.config)
        trainer.restore(checkpoint_path)

        # Assigning extracted policies to focal players
        agents = []
        if self.use_policy_sharing:  # Duplicates the policy across all focal players (universaliaztion)
            for i in range(self.no_of_focal_players):
                agents.append(utils.RayModelPolicy(trainer, "av"))
        else:  # Distribute unique policies in a round-robin fashion
            for i in range(self.no_of_focal_players):  
                agents.append(utils.RayModelPolicy(trainer, f"player_{i}"))

        return agents


    def run(self):
        """ Run the experiment and return the results (run time and rewards)
        Returns:
            A dictionary containing the run time and rewards
        """

        # 1. LOAD THE AGENT POLICIES
        print(f"[INFO] Testing {self.algorithm} on {self.scenario_name}")
        print(f"[INFO] Loading experiment...")
        agents = self.load_experiment()
        print("[INFO] Experiment loaded!")

        # 2. INITIALIZE THE AGENTS
        timestep = self.built_scenario.reset()
        states = [agent.initial_state() for agent in agents] # initialising states
        focal_actions = [0] * len(agents)  # Initializing action list 

        # 3. RUN THE TESTING SCENARIO
        print("[INFO] Running scneario...")
        start = time.time()
        for _ in range(self.config["horizon"]):  # Test over 1 episode (1000 timesteps)

            for i, agent in enumerate(agents):
                # Characterizing the timestep for the agent
                timestep_agent = dm_env.TimeStep(
                    step_type=timestep.step_type,
                    reward=timestep.reward[i],
                    discount=timestep.discount,
                    observation=timestep.observation[i]
                )

                # Each focal agent takes their respective action at this timestep (agent step)
                focal_actions[i], states[i] = agent.step(timestep=timestep_agent, prev_state=states[i])

            # Stepping through the environment with all agent actions (environment step)
            focal_timestep, background_timestep = self.built_scenario.step(focal_actions)

            # Appending the rewards to the rewards dictionary
            self.rewards['Focal rewards'].append(focal_timestep.reward)
            self.rewards['Background rewards'].append(background_timestep.reward)

        end = time.time()
        print("[INFO] Scenario run complete!")
        
        return (end - start), self.rewards
        

def main():

    # Command line arguments
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario_name",
        type=str,
        default='prisoners_dilemma_in_the_matrix_2',
        help="The name of the scenario to test. Must be in scenario.SCENARIOS_BY_SUBSTRATE"
        "Clean Up scenarios take the form: 'clean_up_<0-6>'"
        "Prisoner's Dilemma scenarios take the form: 'prisoners_dilemma_in_the_matrix_<0-5>'")
    parser.add_argument(
        "--algorithm",
        type=str,
        default='PPO',
        help="The name of the algorithm to load and test."
        "Should be one of: 'A3C', 'PPO', or 'R2D2'")
    parser.add_argument(
        "--use_policy_sharing",
        type=honours_utils.boolean_string,
        default=False,
        help="Whether to use policy sharing (True / False)." 
        "It is recommended to mirror the setting used during training.")
    parser.add_argument(
        "--substrate",
        type=str,
        default='prisoners_dilemma_in_the_matrix',
        help="The name of the base substrate to test (clean_up / prisoners_dilemma_in_the_matrix)")
    parser.add_argument(
        "--experiment_path",
        type=str,
        default='testing/pd_ray_logs/PD_PPO_NPS',
        help="The path to the experiment directory where you saved the trained model.")
    
    args = parser.parse_args()

    # Running the testing scenario with the provided arguments
    scenario_tester = ScenarioTesting(
        scenario_name=args.scenario_name, 
        algorithm=args.algorithm, 
        use_policy_sharing=args.use_policy_sharing,
        substrate=args.substrate, 
        experiment_path=args.experiment_path)

    run_time, rewards = scenario_tester.run()
    
    # Displaying the results (run time and custom social metrics)
    print("\n\nElapsed time for performing test: {:.2f}s".format(run_time))
    Social_Metrics(rewards=rewards).report


if __name__ == "__main__":
    main()