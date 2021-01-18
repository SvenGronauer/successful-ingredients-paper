import os
import argparse
import numpy as np
import torch
from itertools import product
from sipga.common.mp_utils import Scheduler, Task
import pybullet_envs  # noqa
import time
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
    def random_bool():  # decide between True and False with 50-50 probability
        return bool(np.random.rand(1) >= 0.5)

    def get_random_dict(lr, train_iters):
        return dict(
            adv_estimation_method='plain',
            epochs=312,  # 9.98M total steps
            train_pi_iterations=train_iters,
            # pi_lr=np.random.uniform(low=lrs[0], high=lrs[1]),
            optimizer='Adam',
            pi_lr=lr,
            use_entropy=random_bool(),
            use_exploration_noise_anneal=random_bool(),
            target_kl=0.00,  # not investigated in this experiment
            use_kl_early_stopping=False,
            use_linear_lr_decay=random_bool(),
            use_max_grad_norm=random_bool(),
            use_mini_batches=False,
            use_reward_scaling=random_bool(),
            use_shared_weights=False,
            use_standardized_advantages=random_bool(),
            use_standardized_obs=random_bool()
        )

    # def loguniform(low=0, high=1, size=None):
    #     log_low = np.log(low)
    #     log_high = np.log(high)
    #     return np.exp(np.random.uniform(log_low, log_high, size))

    # search over the pi learning rates
    lrs = [1e-4, 2.5e-4, 5.0e-4, 1e-3]
    # # mbs = [2*1000, 4*1000, 8*1000]
    pi_iters = [10, 20, 40, 80]
    runs = range(number_of_runs)   # fixed: search over 16 seeds, each ingredient is
    # on average activated 8 times

    for setting in product(lrs, pi_iters, runs):
        lr, iters, _ = setting
        yield get_random_dict(lr=lr, train_iters=iters)


def create_tasks(number_of_runs, log_dir, env_id):
    ts = list()
    alg = 'iwpg'
    defaults = utils.get_defaults_kwargs(alg=alg, env_id=env_id)
    task_number = 0
    gen = hyper_parameter_generator(number_of_runs=number_of_runs)
    hms_time = time.strftime("%Y-%m-%d__%H-%M-%S")

    try:
        while True:
            generated_params = next(gen)
            task_number += 1
            kwargs = defaults.copy()
            experiment_path = os.path.join('experiments_a', env_id)
            logger_kwargs = setup_logger_kwargs(base_dir=log_dir,
                                                exp_name=experiment_path,
                                                seed=task_number,
                                                hms_time=hms_time,
                                                use_tensor_board=False,
                                                verbose=False)
            kwargs.update(logger_kwargs=logger_kwargs,
                          seed=task_number,
                          env_id=env_id,
                          alg=alg,
                          **generated_params)
            target_fn = run_training
            t = Task(id=task_number, target_function=target_fn, kwargs=kwargs)
            ts.append(t)

    except StopIteration:
        print(f'Created {task_number} tasks.')

    return ts


if __name__ == '__main__':

    n_cpus = os.cpu_count()
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--env', type=str, required=True,
                        help='Example: HalfCheetahBulletEnv-v0')
    # parser.add_argument('--search', action='store_true',
    #                     help='If given search over learning rates.')
    parser.add_argument('--num-cores', '-c', type=int, default=n_cpus,
                        help='Number of parallel processes generated.')
    parser.add_argument('--num-runs', '-r', type=int, default=16,
                        help='Number of total runs that are executed.')
    parser.add_argument('--log-dir', type=str, default='/var/tmp/',
                        help='Define a custom directory for logging.')
    args = parser.parse_args()

    scheduler = Scheduler(num_cores=args.num_cores, verbose=False)
    tasks = create_tasks(
        number_of_runs=args.num_runs,
        log_dir=args.log_dir,
        env_id=args.env
    )
    scheduler.fill(tasks)
    scheduler.run()
