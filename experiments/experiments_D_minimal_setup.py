import os
import argparse
import numpy as np
import gym
import time
import torch
from itertools import product
# --- sipga packages
from sipga.common.mp_utils import Scheduler, Task
import pybullet_envs  # noqa
from sipga.common import utils
from sipga.common.loggers import setup_logger_kwargs


# force number of threads to be one to boost performance on Threadripper HW
torch.set_num_threads(1)


def run_training(**kwargs):
    from sipga.common.experiment_analysis import EnvironmentEvaluator
    alg = kwargs.pop('alg')
    env_id = kwargs.pop('env_id')
    logger_kwargs = kwargs.pop('logger_kwargs')
    log_dir = logger_kwargs['log_dir']
    learn_fn = utils.get_learn_function(alg)
    evaluator = EnvironmentEvaluator(log_dir=log_dir)

    ac, env = learn_fn(
        env_id,
        logger_kwargs=logger_kwargs,
        **kwargs
    )
    evaluator.eval(env=env, ac=ac, num_evaluations=32)


def hyper_parameter_generator(number_of_runs):
    """ Generate hyper-parameters for passed number of runs."""

    def get_parameter_dict(lr, train_iters):
        return dict(
            adv_estimation_method='gae',
            epochs=312,  # 9.98M total steps
            train_pi_iterations=train_iters,
            optimizer='RMSprop',
            pi_lr=lr,
            use_entropy=False,
            use_exploration_noise_anneal=False,
            target_kl=0.00,  # not investigated in this experiment
            use_kl_early_stopping=False,
            use_linear_lr_decay=True,
            use_max_grad_norm=False,
            use_reward_scaling=True,
            use_shared_weights=False,
            use_standardized_advantages=False,
            use_standardized_obs=True,
            weight_initialization='kaiming_uniform',
            video_freq=0  # deactivate video recording
        )

    learning_rates = [1e-4, 2.5e-4, 5.0e-4, 1e-3]
    train_iters = [10, 20, 40, 80]
    runs = range(number_of_runs)

    # Third, study parameter sharing
    for setting in product(runs, train_iters, learning_rates):
        _, train_iter, lr = setting
        yield get_parameter_dict(lr=lr, train_iters=train_iter)


def create_tasks(number_of_runs, log_dir, env_id):
    ts = list()
    task_number = 1000  # === seed number
    alg = 'iwpg'

    if env_id == 'all':
        env_ids = ['HalfCheetahBulletEnv-v0',
                   'AntBulletEnv-v0',
                   'HopperBulletEnv-v0',
                   'Walker2DBulletEnv-v0',
                   'HumanoidBulletEnv-v0',
                   'ReacherBulletEnv-v0',
                   'PusherBulletEnv-v0',
                   'KukaBulletEnv-v0',
                   ]
    else:
        env_ids = [env_id, ]

    for env_id in env_ids:
        defaults = utils.get_defaults_kwargs(alg=alg, env_id=env_id)
        gen = hyper_parameter_generator(number_of_runs)

        try:
            while True:
                generated_params = next(gen)
                task_number += 1
                kwargs = defaults.copy()
                experiment_path = os.path.join('experiments_d', env_id)
                logger_kwargs = setup_logger_kwargs(base_dir=log_dir,
                                                    exp_name=experiment_path,
                                                    seed=task_number,
                                                    use_tensor_board=False,
                                                    verbose=False)
                kwargs.update(logger_kwargs=logger_kwargs,
                              seed=task_number,
                              env_id=env_id,
                              alg=alg,
                              **generated_params)
                # deactivate reward scaling for manipulation tasks
                env_type, _ = utils.get_env_type(env_id=env_id)
                if env_type == 'gym_manipulator_envs' or env_type == 'bullet':
                    kwargs['use_reward_scaling'] = False
                target_fn = run_training
                t = Task(id=task_number, target_function=target_fn,
                         kwargs=kwargs)
                ts.append(t)

        except StopIteration:
            pass
    print(f'Created {task_number} tasks.')

    return ts


if __name__ == '__main__':
    n_cpus = os.cpu_count()
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--env', type=str, required=True,
                        help='Example: HalfCheetahBulletEnv-v0')
    parser.add_argument('--num-cores', type=int, default=n_cpus,
                        help='Number of parallel processes generated.')
    parser.add_argument('--num-runs', type=int, default=4,
                        help='Number of total runs that are executed.')
    parser.add_argument('--log-dir', type=str, default='/var/tmp/',
                        help='Define a custom directory for logging.')
    args = parser.parse_args()

    scheduler = Scheduler(num_cores=args.num_cores,
                          verbose=False)
    tasks = create_tasks(number_of_runs=args.num_runs,
                         log_dir=args.log_dir,
                         env_id=args.env)
    scheduler.fill(tasks)
    scheduler.run()
