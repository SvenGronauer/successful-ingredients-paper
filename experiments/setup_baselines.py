""" ================================================================================================
    Source: OpenAi Baselines, https://github.com/openai/baselines/blob/master/baselines/run.py

    A convenient interface to the OpenAI baselines framework.
    Sacred is used to control the experiments by setting the seeds, logging, etc.
    ================================================================================================
"""
import numpy as np
import gym
import json
import sys
from sipga.common import utils

try:
    sys.path.remove(
        '/opt/ros/kinetic/lib/python2.7/dist-packages')  # avoid ROS error on HPCs
except ValueError:
    pass
import re
import os
from collections import defaultdict
from importlib import import_module

from baselines.common.vec_env import VecFrameStack, VecNormalize, VecEnv
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.vec_video_recorder import VecVideoRecorder
from baselines.common.cmd_util import common_arg_parser, parse_unknown_args
from baselines import logger
from baselines.bench import Monitor

MPI = None

all_registered_envs = defaultdict(set)
for env in gym.envs.registry.all():
    try:
        env_type = env.entry_point.split(':')[0].split('.')[-1]
        all_registered_envs[env_type].add(env.id)
    except AttributeError:
        print('Could not fetch ', env.id)

DEBUG = 10
INFO = 20
WARN = 30
ERROR = 40

DISABLED = 50


def make_env(env_id,
             env_type,
             mpi_rank=0,
             subrank=0,
             seed=None,
             parameter_distribution=False,
             training=False,
             reward_scale=1.0,
             gamestate=None,
             flatten_dict_observations=True,
             wrapper_kwargs=None,
             logger_dir=None):
    wrapper_kwargs = wrapper_kwargs or {}
    if env_type == 'atari' or env_type == 'retro':
        raise NotImplementedError
    else:
        env = gym.make(env_id)

    if flatten_dict_observations and isinstance(env.observation_space,
                                                gym.spaces.Dict):
        raise NotImplementedError
        # keys = env.observation_space.spaces.keys()
        # env = gym.wrappers.FlattenDictWrapper(env, dict_keys=list(keys))

    # adjust seed if num_env > 1
    seed = seed + subrank * 1000 if seed is not None else None
    # change seed if env is for evaluation
    seed = seed + int(training) * 10000 if seed is not None else None
    env.seed(seed)
    if parameter_distribution:
        env.enable_parameter_distribution()
    monitor_path = logger_dir and os.path.join(logger_dir,
                                               str(mpi_rank) + '.' + str(
                                                   subrank))
    env = Monitor(env,
                  filename=None,  # =monitor_path  # if logging into .csv files
                  allow_early_resets=True)

    if reward_scale != 1:
        raise NotImplementedError

    return env


def make_vec_env(env_id, env_type, num_env, seed,
                 parameter_distribution,
                 training,
                 wrapper_kwargs=None,
                 start_index=0,
                 reward_scale=1.0,
                 flatten_dict_observations=True,
                 gamestate=None):
    """
    Create a wrapped, monitored SubprocVecEnv for Atari and MuJoCo.
    """
    wrapper_kwargs = wrapper_kwargs or {}
    mpi_rank = MPI.COMM_WORLD.Get_rank() if MPI else 0
    if mpi_rank > 0 and seed is not None:
        print('INFO: changed seed from {} to {} because of MPI.'.format(seed,
                                                                        seed + 100000 * mpi_rank))
    seed = seed + 100000 * mpi_rank if seed is not None else None
    logger_dir = logger.get_dir()

    def make_thunk(rank):
        return lambda: make_env(
            env_id=env_id,
            env_type=env_type,
            mpi_rank=mpi_rank,
            subrank=rank,
            seed=seed,
            parameter_distribution=parameter_distribution,
            training=training,
            reward_scale=reward_scale,
            gamestate=gamestate,
            flatten_dict_observations=flatten_dict_observations,
            wrapper_kwargs=wrapper_kwargs,
            logger_dir=logger_dir
        )

    # set_global_seeds(seed)  # ..Note: Global seeds are set by sacred framework!
    if num_env > 1:  # replace SubprocVecEnv with DummyVencEnv to maintain reproducibility
        return DummyVecEnv(
            [make_thunk(i + start_index) for i in range(num_env)])
    else:
        return DummyVecEnv([make_thunk(start_index)])


def build_env(
        distribution,
        env_id,
        num_env,
        reward_scale,
        seed,
        training,
        use_scaled_rewards=False,
        use_standardized_obs=True
):
    """Creates an environment instance.

    Parameters
    ----------
    args: Namespace
        Holding arguments for Baselines algorithm.
    training: bool
        If True, training mode else evaluation mode.
    distribution: bool
        If True, use a parameter distribution for learning.

    Returns
    -------
    VecEnv
        Nenv parallel instances of the environment.

    """
    # ncpu = multiprocessing.cpu_count()
    # if sys.platform == 'darwin': ncpu //= 2
    # nenv = num_env or ncpu
    env_type, env_id = get_env_type(env_id)

    if env_type in {'atari', 'retro'}:
        raise NotImplementedError
    else:
        # flatten_dict_observations = alg not in {'her'}
        env = make_vec_env(env_id, env_type, num_env, seed,
                           parameter_distribution=distribution,
                           training=training,
                           reward_scale=reward_scale,
                           flatten_dict_observations=False)
        env = VecNormalize(env,
                           ob=use_standardized_obs,
                           ret=use_scaled_rewards,
                           training=training)
    return env


def get_env_type(env_id: str):
    """Determines the type of the environment if there is no args.env_type.

    Parameters
    ----------
    env_id:
        Name of the gym environment.

    Returns
    -------
    env_type: str
    env_id: str
    """
    # Re-parse the gym registry, since we could have new data_driven_control since last time.
    for env in gym.envs.registry.all():
        env_type = env.entry_point.split(':')[0].split('.')[-1]
        all_registered_envs[env_type].add(
            env.id)  # This is a set so add is idempotent

    if env_id in all_registered_envs.keys():
        env_type = env_id
        env_id = [g for g in all_registered_envs[env_type]][0]
    else:
        env_type = None
        for g, e in all_registered_envs.items():
            if env_id in e:
                env_type = g
                break
        if ':' in env_id:
            env_type = re.sub(r':.*', '', env_id)
        assert env_type is not None, 'env_id {} is not recognized in env types'.format(
            env_id,
            all_registered_envs.keys())

    return env_type, env_id


def get_default_network(env_type):
    if env_type in {'atari', 'retro'}:
        return 'cnn'
    else:
        return 'mlp'


def get_alg_module(alg, submodule=None):
    submodule = submodule or alg

    alg_module = import_module('.'.join(['baselines', alg, submodule]))

    return alg_module


def get_learn_function(alg):
    return get_alg_module(alg).learn


def get_learn_function_defaults(alg, env_type):
    alg_defaults = get_alg_module(alg, 'defaults')
    kwargs = getattr(alg_defaults, env_type)()

    return kwargs


# def setup(args):
#     # configure logger, disable logging in child MPI processes (with rank > 0)
#
#     arg_parser = common_arg_parser()
#     if isinstance(args, list):
#         args, unknown_args = arg_parser.parse_known_args(args=args)
#     elif isinstance(args, argparse.Namespace):
#         args, unknown_args = arg_parser.parse_known_args(namespace=args)
#     else:
#         raise TypeError("args: expected to be of type 'list' or 'Namespace'")
#
#     # name_of_all_registered_envs = all_registered_envs
#     extra_args = parse_cmdline_kwargs(unknown_args)
#
#     if MPI is None or MPI.COMM_WORLD.Get_rank() == 0:
#         rank = 0
#         configure_logger(args.log_path)
#     else:
#         rank = MPI.COMM_WORLD.Get_rank()
#         configure_logger(args.log_path, format_strs=[])
#
#     return args, extra_args


def train_with_baselines(
        alg,
        env_id,
        **kwargs
) -> tuple:
    """Run the training of a Baselines algorithm.
    """
    env_type, env_id = get_env_type(env_id)

    learn = get_learn_function(alg)
    print(kwargs)
    alg_kwargs = get_learn_function_defaults(alg, env_type)
    alg_kwargs['network'] = get_default_network(env_type)
    seed = kwargs.pop('seed', None)
    logger_kwargs = kwargs.pop('logger_kwargs')

    # update algorithm's default parameters with parsed kwargs
    for k, v in kwargs.items():
        assert k in alg_kwargs, f'key={k} is contained in: \n{alg_kwargs}'
        alg_kwargs[k] = v

    # normalize obs of all envs
    # rew scaling only in locomotion tasks
    if env_type == 'gym_manipulator_envs' or env_type == 'bullet':
        use_scaled_rewards = False
    else:
        use_scaled_rewards = True

    env = build_env(
        distribution=False,  # parameter dist in env
        env_id=env_id,
        num_env=1,
        reward_scale=alg_kwargs['reward_scale'],
        seed=seed,
        training=True,  # training mode enabled,
        use_scaled_rewards=use_scaled_rewards,
        use_standardized_obs=True
    )

    print('Training {} on {}: {} with arguments \n{}'.format(alg, env_type,
                                                             env_id,
                                                             alg_kwargs))
    model = learn(
        env=env,
        eval_env=None,
        logger_kwargs=logger_kwargs,
        **alg_kwargs
    )
    # unwrapped_env = env.venv.envs[0].unwrapped

    return model, env
