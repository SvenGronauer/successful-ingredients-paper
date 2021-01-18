"""
    Define default parameters for NPG algorithm.
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
        epochs=312,
        gamma=0.99,
        max_ep_len=1000,
        seed=None,
        steps_per_epoch=32 * 1000,  # default: 64k
        target_kl=0.01,
    )


def bullet():
    """ Default hyper-parameters for PyBullet Envs such as KukaBulletEnv-v0."""
    params = defaults()
    params['epochs'] = 312
    params['max_ep_len'] = 1000
    params['pi_lr'] = 1e-4  # default choice is Adam
    params['steps_per_epoch'] = 32 * 1000

    return params


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
