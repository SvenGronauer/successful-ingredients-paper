import os
import argparse
import numpy as np
import tensorflow as tf
from sipga.common.mp_utils import Scheduler, Task
import pybullet_envs  # noqa
from sipga.common.loggers import setup_logger_kwargs
from setup_baselines import train_with_baselines
from itertools import product
from sipga.common.experiment_analysis import EnvironmentEvaluator
import gym


class TFEnvironmentEvaluator(EnvironmentEvaluator):
    def __init__(self, log_dir):
        super().__init__(log_dir)
        self.env = None
        self.model = None

    def eval(self, env, model, num_evaluations):
        self.env = env
        self.model = model

        self.env.training = False

        returns = list()
        ep_lengths = list()
        for i in range(num_evaluations):
            ret, ep_length = self.eval_once()
            returns.append(ret)
            ep_lengths.append(ep_length)

        # now write returns as column into output file...
        self.write_to_output_file(returns)
        print('Saved to:', os.path.join(self.log_dir, self.output_file_name))
        print(f'Mean Ret: { np.mean(returns)} \t'
              f'Mean EpLen: {np.mean(ep_lengths)}')
        self.env.training = True
        return np.array(returns), np.array(ep_lengths)

    def eval_once(self):
        done = False
        x = self.env.reset()
        ret = 0.
        episode_length = 0

        while not done:
            ob = tf.constant(x)
            ac, *_ = self.model.step(ob, training=False)
            ac = ac.numpy().flatten()
            x, r, done, info = self.env.step(ac)
            ret += float(r)
            episode_length += 1
        return ret, episode_length


def run_training(**kwargs):
    log_dir = kwargs['logger_kwargs']['log_dir']
    evaluator = TFEnvironmentEvaluator(log_dir=log_dir)

    model, env = train_with_baselines(
        **kwargs
    )
    evaluator.eval(env=env, model=model, num_evaluations=32)


def create_tasks(alg, number_of_runs, number_of_cores, env_id, log_dir):
    ts = list()

    alg_parse = {
        'trpo': 'trpo_mpi',
        'ppo': 'ppo2'
    }
    assert alg in alg_parse.keys()
    runs = range(number_of_runs)

    if alg == 'ppo':
        task_number = 0
        learning_rates = [0.00025, 0.0005, 0.001, 0.002]
        # train_iters = [5, 25, 50, 100]
        noptepochs = [1, 2, 4, 8]
        for _, lr, num_opt_epochs in product(runs,
                                             learning_rates,
                                             noptepochs):
            task_number += 1
            experiment_path = os.path.join('baselines', env_id, alg)
            logger_kwargs = setup_logger_kwargs(base_dir=log_dir,
                                                exp_name=experiment_path,
                                                seed=task_number,
                                                use_tensor_board=True,
                                                verbose=(number_of_cores == 1))
            kwargs = dict(
                # total_timesteps=3*3200,  # todo: used for debugging
                alg=alg_parse[alg],
                env_id=env_id,
                logger_kwargs=logger_kwargs,
                seed=task_number,
                lr=lr,
                noptepochs=num_opt_epochs,
                nsteps=32000,
                nminibatches=32
            )

            target_fn = run_training
            t = Task(id=task_number, target_function=target_fn, kwargs=kwargs)
            ts.append(t)

    elif alg == 'trpo':
        task_number = 0
        target_kls = [0.01, 0.02, 0.03, 0.05, 0.005]
        for _, target_kl in product(runs, target_kls):
            task_number += 1
            experiment_path = os.path.join('baselines', env_id, alg)
            logger_kwargs = setup_logger_kwargs(base_dir=log_dir,
                                                exp_name=experiment_path,
                                                seed=task_number,
                                                use_tensor_board=True,
                                                verbose=(number_of_cores == 1))
            kwargs = dict(
                alg=alg_parse[alg],
                env_id=env_id,
                logger_kwargs=logger_kwargs,
                seed=task_number,
                max_kl=target_kl
            )

            target_fn = run_training
            t = Task(id=task_number, target_function=target_fn, kwargs=kwargs)
            ts.append(t)

    else:
        raise NotImplementedError(f'Alg={alg}; only ppo and trpo supported')

    return ts


if __name__ == '__main__':
    n_cpus = os.cpu_count()
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--alg', type=str, required=True,
                        help='Choose between: ppo, trpo')
    parser.add_argument('--env', type=str, required=True,
                        help='Example: HalfCheetahBulletEnv-v0')
    parser.add_argument('--search', action='store_true',
                        help='If given search over learning rates.')
    parser.add_argument('--num-cores', type=int, default=n_cpus,
                        help='Number of parallel processes generated.')
    parser.add_argument('--num-runs', type=int, default=4,
                        help='Number of total runs that are executed.')
    parser.add_argument('--log-dir', type=str, default='/var/tmp/',
                        help='Define a custom directory for logging.')
    args = parser.parse_args()

    scheduler = Scheduler(num_cores=args.num_cores,
                          verbose=False)
    tasks = create_tasks(
        alg=args.alg,
        number_of_runs=args.num_runs,
        number_of_cores=args.num_cores,
        env_id=args.env,
        log_dir=args.log_dir
    )
    scheduler.fill(tasks)
    scheduler.run()
