from sipga.common.model import Model
import argparse
import os
import torch
import warnings
try:
    import pybullet_envs  # noqa
except ImportError:
    warnings.warn('WARNING: pybullet_envs package not found. '
                  'Please install PyBullet via pip: pip install pybullet')

# force number of threads to be one to boost performance on Threadripper HW
torch.set_num_threads(1)


if __name__ == '__main__':
    print('Run the training of my sipga repository...')
    n_cpus = os.cpu_count()
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--alg', type=str, required=True,
                        help='Choose from: {iwpg, ppo, trpo, npg}')
    parser.add_argument('--debug', action='store_true',
                        help='Show debug prints during training.')
    parser.add_argument('--env', type=str, required=True,
                        help='Example: HopperBulletEnv-v0')
    parser.add_argument('--play', action='store_true',
                        help='Visualize agent after training.')
    parser.add_argument('--seed', default=0, type=int,
                        help='Define the init seed for experiments.')
    parser.add_argument('--search', action='store_true',
                        help='If given search over learning rates.')
    parser.add_argument('--num-cores', '-c', type=int, default=n_cpus,
                        help='Number of parallel processes generated.')
    parser.add_argument('--num-runs', '-r', type=int, default=1,
                        help='Number of total runs that are executed.')
    parser.add_argument('--log-dir', type=str, default='/var/tmp/',
                        help='Define a custom directory for logging.')
    args = parser.parse_args()

    model = Model(alg=args.alg,
                  env_id=args.env,
                  log_dir=args.log_dir,
                  seed=args.seed)
    model.compile(num_runs=args.num_runs,
                  num_cores=args.num_cores)

    model.fit()
    if args.play:
        model.play()
