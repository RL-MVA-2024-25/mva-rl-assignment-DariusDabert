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


from fast_env_hiv import FastHIVPatient


@dataclass
class DoOnce:
    do: bool = True

    def __call__(self):
        if self.do:
            self.do = False
            return True
        return False


def build_env(domain_randomization: bool):
    env = FastHIVPatient(domain_randomization=domain_randomization)
    env = TransformObservation(
        env,
        lambda obs: np.log(np.maximum(obs, 1e-8)),
    )
    env = TimeLimit(env, max_episode_steps=200)
    return env


def generate_rollouts(
    fixed_environment: bool,
    environment_count: int,
    rollout_count: int,
    random_generator: np.random.Generator,
) -> types.TrajectoryWithRew:
    collected_trajectories = []

    for _ in trange(0, rollout_count, environment_count, desc="Generating rollouts"):
        vectorized_env = make_vec_env(
            lambda *, init_once: build_env(init_once()),
            n_envs=environment_count,
            vec_env_cls=SubprocVecEnv,
            env_kwargs=dict(init_once=DoOnce(fixed_environment)),
        )
        sample_condition = make_sample_until(min_episodes=environment_count)
        trajectory_accumulator = TrajectoryAccumulator()
        trajectory_list = []
        observations = vectorized_env.reset()

        wrapped_observations = types.maybe_wrap_in_dictobs(observations)

        for environment_index, observation in enumerate(wrapped_observations):
            trajectory_accumulator.add_step(dict(obs=observation), environment_index)

        active_environments = np.ones(vectorized_env.num_envs, dtype=bool)

        while np.any(active_environments):
            actions = np.array(
                vectorized_env.env_method(
                    "greedy_action", num_watch_steps=5, consecutive_actions=1
                )
            )

            observations, rewards, dones, infos = vectorized_env.step(actions)

            wrapped_observations = types.maybe_wrap_in_dictobs(observations)
            dones &= active_environments

            new_trajectories = trajectory_accumulator.add_steps_and_auto_finish(
                actions,
                wrapped_observations,
                rewards,
                dones,
                infos,
            )
            trajectory_list.extend(new_trajectories)

            if sample_condition(trajectory_list):
                active_environments &= ~dones

        collected_trajectories.extend(trajectory_list)

    random_generator.shuffle(collected_trajectories)

    processed_trajectories = [dataclasses.replace(traj, infos=None) for traj in collected_trajectories]
    statistics = rollout_stats(processed_trajectories)
    print(f"Rollout statistics: {statistics}")

    return processed_trajectories

def validation_callback(
    trainer: bc.BC,
    environment_count: int,
    experiment_name: str,
    repetition_index: int,
    highest_score: float,
    best_random_env_reward: float,
    best_patient_env_reward: float,
):
    def callback():
        random_env = make_vec_env(
            lambda: build_env(domain_randomization=True),
            n_envs=environment_count,
        )
        patient_env = make_vec_env(
            lambda: build_env(domain_randomization=False),
            n_envs=environment_count,
        )

        random_mean_reward, random_std_reward = evaluate_policy(
            trainer.policy, random_env, n_eval_episodes=20
        )
        patient_mean_reward, patient_std_reward = evaluate_policy(
            trainer.policy, patient_env, n_eval_episodes=10
        )

        print(f"----- Epoch {callback.epoch} - Validation Step -----")
        print(f"Random environment reward: {random_mean_reward:.2e} ± {random_std_reward:.2e}")
        print(f"Patient environment reward: {patient_mean_reward:.2e} ± {patient_std_reward:.2e}")
        print("----------------------------------------")


        score = calculate_score(random_mean_reward, patient_mean_reward)

        if score > callback.highest_score or (
            random_mean_reward > callback.best_random_env_reward
            and patient_mean_reward > callback.best_patient_env_reward
        ):
            save_directory = Path("models/bc") / experiment_name
            save_directory.mkdir(parents=True, exist_ok=True)

            callback.highest_score = score
            callback.best_random_env_reward = random_mean_reward
            callback.best_patient_env_reward = patient_mean_reward

            save_policy(
                trainer.policy,
                save_directory / f"repeat_{repetition_index}_best.pkl",
            )

        callback.epoch += 1

    callback.epoch = 0
    callback.highest_score = highest_score
    callback.best_random_env_reward = best_random_env_reward
    callback.best_patient_env_reward = best_patient_env_reward
    return callback


def calculate_score(random_env_reward: float, patient_env_reward: float):
    score = 0
    if patient_env_reward >= 3432807.680391572:
        score += 1
    if patient_env_reward >= 1e8:
        score += 1
    if patient_env_reward >= 1e9:
        score += 1
    if patient_env_reward >= 1e10:
        score += 1
    if patient_env_reward >= 2e10:
        score += 1
    if patient_env_reward >= 5e10:
        score += 1
    if random_env_reward >= 1e10:
        score += 1
    if random_env_reward >= 2e10:
        score += 1
    if random_env_reward >= 5e10:
        score += 1
    return score

def execute_training(
    rollout_count: int,
    environment_count: int,
    experiment_name: str,
    include_fixed_environment: bool = True,
    computation_device: str = "auto",
    rollout_file_path: Path | None = None,
    training_epochs: int = 5,
    repeat_iterations: int = 1,
):
    random_generator = np.random.default_rng()

    if rollout_file_path is None:
        trajectory_data = generate_rollouts(
            include_fixed_environment, environment_count, rollout_count, random_generator=random_generator
        )
        save_directory = Path("data/rollouts") / (
            f"{experiment_name}_{rollout_count}.traj"
        )
        save_directory.parent.mkdir(parents=True, exist_ok=True)
        serialize.save(save_directory, trajectory_data)
        print(f"Saved rollouts to {save_directory}")
    else:
        trajectory_data = serialize.load(rollout_file_path)
        print(f"Loaded rollouts from {rollout_file_path}")

    flattened_transitions = rollout.flatten_trajectories(trajectory_data)
    randomized_env = make_vec_env(
        lambda: build_env(domain_randomization=True),
        n_envs=environment_count,
    )
    patient_env = make_vec_env(
        lambda: build_env(domain_randomization=False),
        n_envs=environment_count,
    )
    base_environment = FastHIVPatient(domain_randomization=False)

    highest_score = 0
    optimal_random_reward = 0
    optimal_patient_reward = 0

    for iteration in range(repeat_iterations):

        behavior_cloning_trainer = bc.BC(
            observation_space=base_environment.observation_space,
            action_space=base_environment.action_space,
            demonstrations=flattened_transitions,
            rng=random_generator,
            device=computation_device,
            custom_logger=(
                configure_logger(
                    folder=Path("logs") / experiment_name, format_strs=["log"]
                )
            ),
        )

        validation_handler = validation_callback(
            behavior_cloning_trainer,
            environment_count,
            experiment_name,
            iteration,
            highest_score,
            optimal_random_reward,
            optimal_patient_reward,
        )

        behavior_cloning_trainer.train(
            n_epochs=training_epochs,
            on_epoch_end=validation_handler,
        )

        avg_random_reward, std_random_reward = evaluate_policy(
            behavior_cloning_trainer.policy, randomized_env, n_eval_episodes=10
        )
        avg_patient_reward, std_patient_reward = evaluate_policy(
            behavior_cloning_trainer.policy, patient_env, n_eval_episodes=10
        )

        print(f"Average Random Reward: {avg_random_reward:.2e} ± {std_random_reward:.2e}")
        print(f"Average Patient Reward: {avg_patient_reward:.2e} ± {std_patient_reward:.2e}")

        highest_score = validation_handler.highest_score
        optimal_random_reward = validation_handler.best_random_env_reward
        optimal_patient_reward = validation_handler.best_patient_env_reward

        final_model_path = Path("models/bc") / experiment_name / f"final_{iteration}_{rollout_count}.pkl"
        final_model_path.parent.mkdir(parents=True, exist_ok=True)
        save_policy(behavior_cloning_trainer.policy, final_model_path)

    print(f"Final model saved at {final_model_path}")


def initiate_training():
    parser = ArgumentParser()
    parser.add_argument("--rollout-count", type=int, default=10000, help="Number of rollouts to generate.")
    parser.add_argument("--environment-count", type=int, default=10, help="Number of parallel environments.")
    parser.add_argument(
        "--disable-fixed-environment", action="store_false", dest="use_fixed_environment",
        help="Disable using a fixed environment for training."
    )
    parser.add_argument("--device", type=str, default="cpu", help="Device to use for training (e.g., 'cpu', 'cuda').")
    parser.add_argument(
        "--rollout-file", "-p", type=Path, default=None,
        help="Path to pre-generated rollout file (optional)."
    )
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--repeats", type=int, default=3, help="Number of training repetitions.")
    parser.add_argument("--experiment-name", default=generate_slug(2), help="Name of the experiment.")
    args = parser.parse_args()

    experiment_name = f"{int(time.time())}_{args.experiment_name}"
    print(f"Starting experiment: {experiment_name}")

    execute_training(
        rollout_count=args.rollout_count,
        environment_count=args.environment_count,
        experiment_name=experiment_name,
        include_fixed_environment=args.use_fixed_environment,
        computation_device=args.device,
        rollout_file_path=args.rollout_file,
        training_epochs=args.epochs,
        repeat_iterations=args.repeats,
    )

# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!

SAVE_PATH = Path(__file__).parent.parent /"models"
MODEL_NAME = "bc/"


class ProjectAgent:
    def act(self, observation, use_random=False):
        observation = np.log(np.maximum(observation, 1e-8))
        return self.policy.predict(observation, deterministic=True)[0]

    def save(self, path):
        torch.save(self.policy, path)

    def load(self):
        self.policy = torch.load(SAVE_PATH / MODEL_NAME, weights_only=False)

if __name__ == "__main__":
    initiate_training()
