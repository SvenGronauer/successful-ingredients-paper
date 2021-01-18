""" Grid Search over Hyper-parameters.

    The parameter grids are in correspondence with SciKit Learn.
    usage:
    python -m sipga.search --env AntBulletEnv-v0 --alg iwpg
    --grid '{"pi_lr": [0.01, 0.001], "gamma": [0.95, 0.9, 0.8]}'

    Author: ---
"""
from sipga.common.loggers import setup_logger_kwargs
from sipga.common import mp_utils, model
import torch
import os
import argparse
import json
from itertools import product
import warnings

try:
    import pybullet_envs  # noqa
except ImportError:
    warnings.warn('pybullet_envs package not found.')


# force number of threads to be one to boost performance on Threadripper HW
torch.set_num_threads(1)


class GridSearch(model.Model):

    def __init__(self,
                 param_grid: str,
                 alg: str,
                 env_id: str,
                 log_dir: str,
                 seed: int
                 ) -> None:
        """ Class Constructor  """
        super().__init__(alg, env_id, log_dir, seed)
        # passed_param_grid = json.loads(param_grid)
        self.param_grid = json.loads(param_grid)

    def _fill_scheduler(self, target_fn):
        """ Create tasks for multi-process execution.

        will be called if model.compile(multi_thread=True) is enabled."""

        ts = list()
        task_number = 1

        # for param_set in self.param_grid:
        for param_set in product(*self.param_grid.values()):
            grid_kwargs = dict(zip(self.param_grid.keys(), param_set))

            for i in range(self.num_runs):
                kwargs = self.kwargs.copy()
                _seed = task_number + self.seed
                logger_kwargs = setup_logger_kwargs(base_dir=self.log_dir,
                                                    exp_name=self.exp_name,
                                                    seed=_seed,
                                                    use_tensor_board=True,
                                                    verbose=False)
                kwargs.update(logger_kwargs=logger_kwargs,
                              seed=_seed,
                              alg=self.alg,
                              env_id=self.env_id)
                # now pass the grid search parameters...
                kwargs.update(**grid_kwargs)
                t = mp_utils.Task(id=_seed,
                                  target_function=target_fn,
                                  kwargs=kwargs)
                ts.append(t)
                task_number += 1

        self.scheduler.fill(tasks=ts)

    def summary(self):
        pass


if __name__ == '__main__':
    n_cpus = os.cpu_count()
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--alg', type=str, required=True,
                        help='Choose from: {ppo, trpo}')
    parser.add_argument('--env', type=str, required=True,
                        help='Example: HopperBulletEnv-v0')
    parser.add_argument('--grid', type=str, required=True,
                        help='Example: `{"pi_lr": [0.01, 0.001], "gamma": [0.95, 0.9, 0.8]}`')
    parser.add_argument('--seed', default=0, type=int,
                        help='Define the init seed for experiments.')
    parser.add_argument('--num-cores', '-c', type=int, default=n_cpus,
                        help='Number of parallel processes generated.')
    parser.add_argument('--num-runs', '-r', type=int, default=8,
                        help='Number of runs per parameter setting.')
    parser.add_argument('--log-dir', type=str, default='/var/tmp',
                        help='Define a custom directory for logging.')
    args = parser.parse_args()

    # ---- Usage ---
    # python -m sipga.search --env AntBulletEnv-v0 --alg ppo
    # --grid '{"pi_lr": [0.01, 0.001], "gamma": [0.95, 0.9, 0.8]}'
    print(args.seed)
    model = GridSearch(
        args.grid,
        alg=args.alg,
        env_id=args.env,
        log_dir=args.log_dir,
        seed=args.seed
    )
    model.compile(num_runs=args.num_runs,
                  num_cores=args.num_cores)
    model.fit()
