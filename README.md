# MARL in Social Dilemmas

*An empirical evaluation of various reinforcement learning algorithms in social dilemmas*

## About

Social dilemmas, characterized by tension between individual and
collective benefit, are important aspects of modern society. Social
psychologists have studied social dilemmas through real-world
empirical analysis. However, due to numerous confounding
variables, there has been a recent shift to computational methods.
Reinforcement learning, particularly multi-agent reinforcement
learning (MARL), offers an avenue to study social dilemmas in a
simulated manner allowing more control over handcrafted
environments. Whilst in its infancy, there has been limited research
in determining the efficacy of various multi-agent algorithms in
studying social dilemmas. This work compares and contrasts three
MARL algorithms (R2D2, PPO, A3C) to determine their efficacy
in solving the Clean Up public goods dilemma and the scaled
canonical Prisoner’s commons dilemma. Through empirical
evaluation, we find that that PPO and R2D2 are the best performing
algorithms in Clean Up and Prisoner’s Dilemma respectively. This
indicates that the efficacy of an algorithm is dependent on the
environment. Furthermore, we provide supporting evidence that
training and evaluation performance alone is insufficient to justify
the selection of an algorithm, promoting an empirical investigation
of algorithm efficacies in future social dilemma research.

This software is the means by which these results/conclusions were
obtained. This research was conducted using the [Melting Pot](https://github.com/deepmind/meltingpot)
protocol, which provides a standardised way to compare and contrast the
efficacy of algorithms across a plethora of high quality environment implementaions.


## Installation

In short, this software requires the user to install [Melting Pot](https://github.com/deepmind/meltingpot) and merge the contents
of this project.

There are 2 main ways to go about this:

1. Use a Devcontainer (x86 only) and merge the additions and changes.

2. Build the required libraries from source and merge the additions and changes.

### Devcontainer

Melting Pot provides a pre-configured development environment ([devcontainer](https://containers.dev)).
This dev container is built using the `Dockerfile` and a `devcontainer.json` files within the ".devcontainer" folder.
Note: The following steps assumes that you already have [Docker](https://www.docker.com/products/docker-desktop/) installed on your system.

1. Download Melting Pot v1.0.4. You can do this in two ways:

    a) Download from the [releases page](https://github.com/deepmind/meltingpot/releases/tag/v1.0.4) and extract the contents.
    b) Clone the repository into container using the devcontainer [VSCode extension](https://code.visualstudio.com/docs/remote/containers-tutorial).
    This should fetch the latest version, being version 1.0.4 at the time of writing this, and automatically launch a containerised environment.

2. Launch the devcontainer using the [VSCode Containers](https://code.visualstudio.com/docs/remote/containers-tutorial) extension.
Note: You can skip this step if you followed option 1.b, as this would have already been done for you.

Open the extracted folder in VSCode. The devcontainer extension should automatically detect the ".devcontainer" subdirectory inside the project and
should subsequetly promompt you to run a container for the project. Accept this and wait for the process to finish. Note: This process will be shorter
the next time you run the container.

3. Merge the additions and changes made by my software.

Simply drag and drop all the contents from my submission into the top-level directory (i.e. the same level as the READ_ME.md file). When promoted, give permission to
override any files. Your software should now contain two new folders: "honoursproj" and "testing", along with the modifications I have made to some of the Melting Pot source files.
You should be able to view/track these changes using git.

4. Download the models

You will need to download 2 sets of models. The first are the Melting Pot models for the background agents which is required if you run any 'scenario tests'. The
second are the models which I trained for this research, which will govern the focal players' policies.

Download the [Melting Pot models](https://storage.googleapis.com/dm-meltingpot/meltingpot-assets-1.0.0.tar.gz) and extract the contents to meltingpot/assets/saved_models.
Note, if you chose 1.a, this this might have already been done for you.

Download my trained A3C, PPO, and R2D2 models from this [Google Drive link](https://drive.google.com/file/d/1Go5dJ2q7AWLE7fFu8C4QFwrPd8NIdWR7/view?usp=share_link). 
Extract the contents to the "testing" subdirectory (from step 3). You should now have two subfolders in the "testing" folder: "cu_ray_logs" and "pd_ray_logs" which
contain the A3C, PPO, and R2D2 models for Clean Up and Prisoner's Dilemma respectively.

5. (Optional) Install `ffmpeg` for environment recording

    On Unbuntu:
    ```shell
    sudo apt-get install ffmpeg
    ```


#### CUDA support
To enable CUDA support (required for GPU training), make sure you have the
[nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) package installed. You will also have to enable gpus in
in `devcontainer.json` file by adding the `--gpus all` flag under the `runArgs` grouped parameters. the *devcontainer* extension will prompt you to relauch
the container with the new parameter.


### Manual install
These manual installation instructions follow the procedure laid out by [Melting Pot](https://github.com/deepmind/meltingpot)

NOTE: If you encounter a `NotFound`-esque error during any of the installation options, you can solve this by exporting the meltingpot home directory as `PYTHONPATH` (e.g. by calling `export PYTHONPATH=$(pwd)`).

1.  (Optional) Activate a virtual environment, e.g.:

    ```shell
    python3 -m venv "${HOME}/meltingpot_venv"
    source "${HOME}/meltingpot_venv/bin/activate"
    ```

2.  Install `dmlab2d` from the
    [dmlab2d wheel files](https://github.com/deepmind/lab2d/releases/tag/release_candidate_2022-03-24), e.g.:

    ```shell
    pip3 install https://github.com/deepmind/lab2d/releases/download/release_candidate_2022-03-24/dmlab2d-1.0-cp39-cp39-manylinux_2_31_x86_64.whl
    ```

    If there is no appropriate wheel (e.g. M1 chipset) you will need to install
    [`dmlab2d`](https://github.com/deepmind/lab2d) and build the wheel yourself
    (see
    [`install.sh`](https://github.com/deepmind/meltingpot/blob/main/install.sh)
    for an example installation script that can be adapted to your setup).

3.  Test the `dmlab2d` installation in `python3`:

    ```python
    import dmlab2d
    import dmlab2d.runfiles_helper

    lab = dmlab2d.Lab2d(dmlab2d.runfiles_helper.find(), {"levelName": "chase_eat"})
    env = dmlab2d.Environment(lab, ["WORLD.RGB"])
    env.step({})
    ```

4.  Install Melting Pot:

    ```shell
    git clone -b main https://github.com/deepmind/meltingpot
    cd meltingpot
    curl -L https://storage.googleapis.com/dm-meltingpot/meltingpot-assets-1.0.0.tar.gz \
        | tar -xz --directory=meltingpot
    pip3 install .
    ```

5.  Test the Melting Pot installation:

    ```shell
    pip3 install pytest
    pytest meltingpot
    ```

6. Install the RLlib dependencies:
    
    ```shell
    cd <meltingpot_root>
    pip3 install -e .[rllib]
    ```

7. Merge the additions and changes made by my software.

Simply drag and drop all the contents from my submission into the top level directory (i.e. the same level as the READ_ME.md file). When promoted, give permission to
override any files. Your software shoud now contain two new folders: "honoursproject" and "testing", along with the modifications I have made to some of the Melting Pot source files.
You should be able to view/track these changes using git.

8. Download the models

You will need to download 2 sets of models. The first are the Melting Pot models for the background agents which is required if you run any 'scenario tests'. The
second are the models which I trained for this research, which will govern the focal players' policies.

Download the [Melting Pot models](https://storage.googleapis.com/dm-meltingpot/meltingpot-assets-1.0.0.tar.gz) and extract the contents to meltingpot/assets/saved_models.
Note, if you chose 1.a, this this might have already been done for you.

Download my trained A3C, PPO, and R2D2 models from this [Google Drive link](https://drive.google.com/file/d/1Go5dJ2q7AWLE7fFu8C4QFwrPd8NIdWR7/view?usp=share_link). 
Extract the contents to the "testing" subdirectory (from step 3). You should now have two subfolders in the "testing" folder: "cu_ray_logs" and "pd_ray_logs" which
contain the A3C, PPO, and R2D2 models for Clean Up and Prisoner's Dilemma respectively.

9. (Optional) Install `ffmpeg` for environment recording

    On Unbuntu:
    ```shell
    sudo apt-get install ffmpeg
    ```

## Usage
This software is comprised of two main scripts: 

1. `training_and_eval.py` - runs the training and evaluation of a substrate using a particular algorithm.

2. `scenario_testing.py` - tests the trained focal policies in a testing scenario and reports back on various custom social metrics.

Each of these scripts are designed to be run from the command line, as is the standard for SOTA approaches.

`training_and_eval.py` takes the following command line arguments:

- `-- experiment_name`: A custom name for the experiment.

- `-- substrate`: The substrate to use for training and evaluation ('clean_up' or 'prisoners_dilemma_in_the_matrix'). 

- `-- algorithm`: The algorithm to use for training and evaluation ('A3C', 'PPO', 'R2D2'). 

- `-- use_policy_sharing`: Whether to use policy sharing or not (True / False).

- `-- num_iterations`: The number of iterations to train for.

- `-- local_dir`: The directory to store the experiment results in.

- `-- use_gpu`: Whether to use GPU training or not (True / False). 

- `-- record_env`: Whether to record the environment or not (True / False).

Example usage: 
    
```shell
python honoursproj/train_and_eval.py --experiment_name CustomExp --substrate clean_up --algorithm A3C --use_policy_sharing True --num_iterations 10 --local_dir testing/temp --use_gpu False --record_env True
```

`scenario_testing.py` takes the following command line arguments:

- `--scenario_name`: The name of the scenario to test. Must be in scenario.SCENARIOS_BY_SUBSTRATE.
    Clean Up scenarios take the form: 'clean_up_<0-6>'
    Prisoner's Dilemma scenarios take the form: 'prisoners_dilemma_in_the_matrix_<0-5>'

- `--algorithm`: The name of the algorithm to load and test.

- `--use_policy_sharing`: (True / False) Whether to use policy sharing.
    It is recommended to mirror the setting used during training.

- `--substrate`: The name of the base substrate to test (clean_up / prisoners_dilemma_in_the_matrix)

- `--experiment_path`: The path to the experiment directory where you saved the trained model.

Example usage:

```shell
python honoursproj/scenario_testing.py --scenario_name prisoners_dilemma_in_the_matrix_2 --algorithm PPO --use_policy_sharing False --substrate prisoners_dilemma_in_the_matrix --experiment_path testing/pd_ray_logs/PD_PPO_NPS
```

## Documentation
For a software report, including toolkit evaluation and usability strategies, refer to the [docs subfolder](/docs)

## Citing Research
If you intend to use these findings in your work, please cite my accompanying thesis:

```bibtex
@inproceedings{sashen2022marlsocialdilemmas,
    title={Multi-Agent Reinforcement Learning in Social Dilemmas},
    author={Sashen Moodley},
    year={2022},
    organization={UKZN}
}
```
