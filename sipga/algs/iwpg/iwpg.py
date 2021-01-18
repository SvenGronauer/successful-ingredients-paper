""" Simple PyTorch implementation of Importance-weighted Policy Gradient (IWPG)

Author:     Still Anonymous
based on:   Spinning Up's Repository
            https://spinningup.openai.com/en/latest/
"""
import numpy as np
import gym
import time
import torch
import torch.optim
from sipga.algs import core
from sipga.common import loggers
import os


class ImportanceWeightedPolicyGradientAlgorithm(core.Algorithm):
    def __init__(
            self,
            actor: str,
            ac_kwargs: dict,
            env_id: str,
            epochs: int,
            logger_kwargs: dict,
            adv_estimation_method: str = 'gae',
            entropy_coef: float = 0.01,
            gamma: float = 0.99,
            lam: float = 0.95,  # GAE scalar
            max_ep_len: int = 1000,
            max_grad_norm: float = 0.5,
            num_mini_batches: int = 16,
            optimizer: str = 'Adam',  # policy optimizer
            optimizer_eps: float = 1e-8,  # improve numerical stability
            pi_lr: float = 3e-4,
            reward_scale: float = 1.0,
            steps_per_epoch: int = 32*1000,
            subtract_adv_mean: bool = True,
            target_kl: float = 0.01,
            train_pi_iterations: int = 80,
            train_v_iterations: int = 5,
            trust_region='plain',  # used for easy filtering in plot utils
            use_entropy: bool = False,
            use_exploration_noise_anneal: bool = False,
            use_kl_early_stopping: bool = False,
            use_linear_lr_decay: bool = True,
            use_max_grad_norm: bool = False,
            use_reward_scaling: bool = True,
            use_shared_weights: bool = False,
            use_standardized_advantages: bool = False,
            use_standardized_obs: bool = True,
            vf_lr: float = 1e-3,
            weight_initialization: str = 'kaiming_uniform',
            save_freq: int = 10,
            seed=None,
            verbose: bool = False,
            **kwargs  # use to log parameters from child classes
    ):
        assert train_pi_iterations > 0
        assert train_v_iterations > 0
        self.alg = ''
        self.entropy_coef = entropy_coef if use_entropy else 0.0
        self.epochs = epochs
        self.max_ep_len = max_ep_len
        self.max_grad_norm = max_grad_norm
        self.num_mini_batches = num_mini_batches
        self.pi_lr = pi_lr
        self.reward_scale = reward_scale
        self.save_freq = save_freq
        self.steps_per_epoch = steps_per_epoch
        self.subtract_adv_mean = subtract_adv_mean
        self.target_kl = target_kl
        self.train_pi_iterations = train_pi_iterations
        self.train_v_iterations = train_v_iterations
        self.use_exploration_noise_anneal = use_exploration_noise_anneal
        self.use_kl_early_stopping = use_kl_early_stopping
        self.use_linear_lr_decay = use_linear_lr_decay
        self.use_max_grad_norm = use_max_grad_norm
        self.use_reward_scaling = use_reward_scaling
        self.verbose = verbose
        self.vf_lr = vf_lr

        # Set up logger and save configuration
        self.params = locals()  # get before logger instance to avoid unnecessary prints
        self.params.pop('self')  # pop to avoid self object errors
        # move nested kwargs to highest dict level
        if 'kwargs' in self.params:
            _kw_args = self.params.pop('kwargs')
            self.params.update(**_kw_args)
        self.logger = loggers.EpochLogger(**logger_kwargs)
        if verbose:
            self.logger.log(f'Run PolicyGradientAlgorithm with kwargs:', 'yellow')
            print(self.params)
        self.logger.save_config(self.params)

        # Instantiate environment
        self.env = env = gym.make(env_id) if isinstance(env_id, str) else env_id
        assert isinstance(env, gym.Env), 'Env is not the expected type.'
        obs_dim = env.observation_space.shape
        act_dim = env.action_space.shape
        # Collect information from environment if it has an time wrapper
        if hasattr(self.env, '_max_episode_steps'):
            self.max_ep_len = self.env._max_episode_steps

        if seed:
            torch.manual_seed(seed)
            np.random.seed(seed)
            self.env.seed(seed=seed)

        self.ac = core.ActorCritic(
            actor_type=actor,
            observation_space=env.observation_space,
            action_space=env.action_space,
            use_standardized_obs=use_standardized_obs,
            use_scaled_rewards=use_reward_scaling,
            use_shared_weights=use_shared_weights,
            weight_initialization=weight_initialization,
            ac_kwargs=ac_kwargs
        )

        # Set up experience buffer
        self.buf = core.Buffer(
            actor_critic=self.ac,
            obs_dim=obs_dim,
            act_dim=act_dim,
            size=steps_per_epoch,
            gamma=gamma,
            lam=lam,
            adv_estimation_method=adv_estimation_method,
            use_scaled_rewards=use_reward_scaling,
            standardize_env_obs=use_standardized_obs,
            standardize_advantages=use_standardized_advantages,
            subtract_advantage_mean=subtract_adv_mean
        )

        # Set up optimizers for policy and value function
        self.pi_optimizer = core.get_optimizer(optimizer, module=self.ac.pi,
                                               lr=pi_lr, eps=optimizer_eps)
        self.vf_optimizer = core.get_optimizer('Adam', module=self.ac.v,
                                               lr=vf_lr)

        # setup scheduler for policy learning rate decay
        self.scheduler = None
        if use_linear_lr_decay:
            def lm(epoch): return 1 - epoch / self.epochs  # linear anneal
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer=self.pi_optimizer,
                lr_lambda=lm
            )

        # Set up model saving
        self.logger.setup_torch_saver(self.ac)
        self.logger.torch_save()

        # setup statistics
        self.start_time = time.time()
        self.epoch_time = time.time()
        self.loss_pi_before = 0.0
        self.loss_v_before = 0.0

    def algorithm_specific_logs(self):
        """ Use this method to collect log information. """
        pass

    def compute_loss_pi(self, data: dict) -> tuple:
        # Policy loss
        dist, _log_p = self.ac.pi(data['obs'], data['act'])
        ratio = torch.exp(_log_p - data['log_p'])

        loss_pi = -(ratio * data['adv']).mean()
        loss_pi -= self.entropy_coef * dist.entropy().mean()

        # Useful extra info
        approx_kl = (0.5 * (dist.mean - data['act']) ** 2
                     / dist.stddev ** 2).mean().item()
        ent = dist.entropy().mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, ratio=ratio.mean().item())

        return loss_pi, pi_info

    # Set up function for computing value loss
    def compute_loss_v(self, obs, ret):
        return ((self.ac.v(obs) - ret) ** 2).mean()

    def learn(self) -> tuple:
        # Main loop: collect experience in env and update/log each epoch
        for epoch in range(self.epochs):
            self.epoch_time = time.time()

            if self.use_exploration_noise_anneal:  # update internals of AC
                self.ac.update(frac=epoch / self.epochs)

            self.roll_out()  # collect data and store to buffer

            # Save model
            if (epoch % self.save_freq == 0) or (epoch == self.epochs - 1):
                self.logger.save_state(state_dict={}, itr=None)

            # Perform policy and value function updates
            self.update()

            # Save information about epoch
            self.log(epoch)

        return self.ac, self.env

    def log(self, epoch: int):
        # Log info about epoch
        total_env_steps = (epoch + 1) * self.steps_per_epoch
        fps = self.steps_per_epoch/(time.time() - self.epoch_time)
        if self.scheduler and self.use_linear_lr_decay:
            current_lr = self.scheduler.get_last_lr()[0]
            self.scheduler.step()  # step the scheduler if provided
        else:
            current_lr = self.pi_lr

        self.logger.log_tabular('Epoch', epoch+1)
        self.logger.log_tabular('EpRet', min_and_max=True, std=True)
        self.logger.log_tabular('EpLen', min_and_max=True)
        self.logger.log_tabular('Values/V', min_and_max=True)
        self.logger.log_tabular('Values/Adv', min_and_max=True)
        self.logger.log_tabular('Loss/Pi', std=False)
        self.logger.log_tabular('Loss/Value')
        self.logger.log_tabular('Loss/DeltaPi')
        self.logger.log_tabular('Loss/DeltaValue')
        self.logger.log_tabular('Entropy')
        self.logger.log_tabular('KL')
        self.logger.log_tabular('Misc/StopIter')
        self.logger.log_tabular('PolicyRatio')
        self.logger.log_tabular('LR', current_lr)
        if self.use_reward_scaling:
            reward_scale_mean = self.ac.ret_oms.mean.item()
            reward_scale_stddev = self.ac.ret_oms.std.item()
            self.logger.log_tabular('Misc/RewScaleMean', reward_scale_mean)
            self.logger.log_tabular('Misc/RewScaleStddev', reward_scale_stddev)
        # some child classes may add information to logs
        self.algorithm_specific_logs()
        self.logger.log_tabular('Time', time.time() - self.start_time)
        self.logger.log_tabular('TotalEnvSteps', total_env_steps)
        self.logger.log_tabular('FPS', fps)

        self.logger.dump_tabular()

    def roll_out(self):
        """collect data and store to experience buffer."""
        o, ep_ret, ep_len = self.env.reset(), 0., 0.

        for t in range(self.steps_per_epoch):
            a, v, logp = self.ac.step(
                torch.as_tensor(o, dtype=torch.float32))

            next_o, r, d, info = self.env.step(a)
            ep_ret += r
            ep_len += 1

            # save and log - Note: reward scaling is performed in buf
            self.buf.store(obs=o, act=a, val=v, logp=logp,
                           rew=r*self.reward_scale)
            self.logger.store(**{
                'Values/V': v})
            o = next_o

            timeout = ep_len == self.max_ep_len
            terminal = d or timeout
            epoch_ended = t == self.steps_per_epoch - 1

            if terminal or epoch_ended:
                if timeout or epoch_ended:
                    _, v, *_ = self.ac(torch.as_tensor(o, dtype=torch.float32))
                else:
                    v = 0.
                self.buf.finish_path(last_val=v)
                if terminal:  # only save EpRet / EpLen if trajectory finished
                    self.logger.store(EpRet=ep_ret, EpLen=ep_len)
                o, ep_ret, ep_len = self.env.reset(), 0., 0.

    def update(self):
        data = self.buf.get()
        self.update_policy_net(data=data)
        self.update_value_net(data=data)

    def update_policy_net(self, data):
        # Get loss and info values before update
        pi_l_old, pi_info_old = self.compute_loss_pi(data)
        self.loss_pi_before = pi_l_old.item()
        self.loss_v_before = self.compute_loss_v(data['obs'],
                                                 data['target_v']).item()
        # get prob. distribution before updates
        p_dist = self.ac.pi.dist(data['obs'])

        # Train policy with multiple steps of gradient descent
        for i in range(self.train_pi_iterations):
            self.pi_optimizer.zero_grad()
            loss_pi, pi_info = self.compute_loss_pi(data=data)
            loss_pi.backward()
            if self.use_max_grad_norm:  # apply L2 norm
                torch.nn.utils.clip_grad_norm_(
                    self.ac.pi.parameters(),
                    self.max_grad_norm)
            self.pi_optimizer.step()
            q_dist = self.ac.pi.dist(data['obs'])
            torch_kl = torch.distributions.kl.kl_divergence(
                p_dist, q_dist).mean().item()
            if self.use_kl_early_stopping and torch_kl > self.target_kl:
                self.logger.log(f'Early stopping - reaching max KL at step={i}')
                break

        # track when policy iteration is stopped; Log changes from update
        self.logger.store(**{
            'Loss/Pi': self.loss_pi_before,
            'Loss/DeltaPi': loss_pi.item() - self.loss_pi_before,
            'Misc/StopIter':  i + 1,
            'Values/Adv': data['adv'].numpy(),
            'Entropy': pi_info['ent'],
            'KL': torch_kl,
            'PolicyRatio': pi_info['ratio']
            })

    def update_value_net(self, data: dict, num_mini_batches: int = 16) -> None:
        # UPDATE: train value network after policy with # mini-batches = 32
        assert self.steps_per_epoch % self.num_mini_batches == 0
        mbs = self.steps_per_epoch // self.num_mini_batches
        indices = np.arange(self.steps_per_epoch)
        val_losses = []
        for _ in range(self.train_v_iterations):
            np.random.shuffle(indices)  # shuffle for mini-batch updates
            # 0 to mini_batch_size with batch_train_size step
            for start in range(0, self.steps_per_epoch, mbs):
                end = start + mbs  # iterate mini batch times
                mb_indices = indices[start:end]
                self.vf_optimizer.zero_grad()
                loss_v = self.compute_loss_v(
                    obs=data['obs'][mb_indices],
                    ret=data['target_v'][mb_indices])
                loss_v.backward()
                val_losses.append(loss_v.item())
                self.vf_optimizer.step()

        self.logger.store(**{
            'Loss/DeltaValue': np.mean(val_losses) - self.loss_v_before,
            'Loss/Value': self.loss_v_before,
        })


def learn(env_id, **kwargs) -> tuple:
    alg = ImportanceWeightedPolicyGradientAlgorithm(
        env_id=env_id,
        **kwargs
    )
    ac, env = alg.learn()

    return ac, env
