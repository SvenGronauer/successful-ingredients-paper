import numpy as np
import gym
import pybullet_envs  # noqa


class EnvRandomEvaluator:
    def __init__(self, env_id):
        self.env = gym.make(env_id)

    def run_once(self):
        ''' Run a single environment for a single episode '''
        # env = gym.make(env_name)
        self.env.reset()
        done = False
        episode_length = 0
        ret = 0.
        while not done:
            _, r, done, _ = self.env.step(self.env.action_space.sample())
            episode_length += 1
            ret += r

        return ret, episode_length
        # print(f'Ran {env_name} for {step} steps.')

    def evaluate_random_policy(self, num_evaluations):
        returns = np.empty((num_evaluations,))
        episode_lengths = np.empty((num_evaluations,))

        for i in range(num_evaluations):
            returns[i], episode_lengths[i] = self.run_once()
            if i % 10 == 0:
                print(i)

        return returns, episode_lengths


if __name__ == '__main__':

    envs = [
        # 'HalfCheetahBulletEnv-v0',
        # 'AntBulletEnv-v0',
        # 'HopperBulletEnv-v0',
        # 'Walker2DBulletEnv-v0',
        # 'HumanoidBulletEnv-v0',
        # 'ReacherBulletEnv-v0',
        # 'PusherBulletEnv-v0',
        # 'ThrowerBulletEnv-v0',
        'KukaBulletEnv-v0',
    ]
    num_evals = 100

    for env_id in envs:
        ev = EnvRandomEvaluator(env_id)
        rets, ep_lens = ev.evaluate_random_policy(num_evals)

        print(
            f'Env: {env_id} Av. Return: {np.mean(rets)} Av. Ep.Length: {np.mean(ep_lens)}')
