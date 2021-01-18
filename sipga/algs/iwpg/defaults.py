"""
    Define default parameters for Importance-weighted Policy Gradient (IWPG)
    algorithm.
"""


def defaults():
    return dict(
        actor='mlp',
        ac_kwargs={
            'pi': {'hidden_sizes': (64, 64),
                   'activation': 'tanh'},
            'val': {'hidden_sizes': (64, 64),
                    'activation': 'tanh'}
        },
        adv_estimation_method='gae',
        seed=None,
        epochs=312,
        gamma=0.99,
        max_ep_len=1000,
        steps_per_epoch=32 * 1000,
    )


def bullet():
    """ Default hyper-parameters for PyBullet Envs such as KukaBulletEnv-v0."""
    return defaults()


def gym_locomotion_envs():
    """Default hyper-parameters for Bullet's locomotion environments."""
    params = defaults()
    params['epochs'] = 312
    params['max_ep_len'] = 1000
    params['pi_lr'] = 1e-4  # default choice is Adam
    params['steps_per_epoch'] = 32 * 1000

    return params


def gym_manipulator_envs():
    params = defaults()
    params['epochs'] = 312
    params['max_ep_len'] = 150
    params['pi_lr'] = 1e-4  # default choice is Adam
    params['steps_per_epoch'] = 32 * 1000

    return params
