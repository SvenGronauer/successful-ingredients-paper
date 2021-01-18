import unittest
import gym
import pybullet_envs  # noqa
import sipga.common.utils as U
from sipga.algs import core
from sipga.common.loggers import setup_logger_kwargs


class TestAlgorithms(unittest.TestCase):

    @staticmethod
    def check_alg(alg_name, env_id):
        """" Run one epoch update with algorithm."""
        print(f'Run {alg_name}.')
        defaults = U.get_defaults_kwargs(alg=alg_name, env_id=env_id)
        defaults['epochs'] = 1
        defaults['num_mini_batches'] = 4
        defaults['steps_per_epoch'] = 1000
        defaults['verbose'] = False
        learn_fn = U.get_learn_function(alg_name)
        defaults['logger_kwargs'] = setup_logger_kwargs(
            exp_name='unittest',
            seed=None,
            base_dir='/var/tmp/',
            datestamp=True,
            use_tensor_board=True,
            verbose=False)
        return learn_fn(env_id, **defaults)

    def test_algorithms(self):
        """ Run all the specified algorithms."""
        for alg in ['iwpg', 'npg', 'trpo', 'ppo']:
            ac, env = self.check_alg(alg, 'HopperBulletEnv-v0')
            self.assertTrue(isinstance(ac, core.ActorCritic))
            self.assertTrue(isinstance(env, gym.Env))


if __name__ == '__main__':
    unittest.main()
