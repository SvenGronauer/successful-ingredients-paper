""" PyTorch implementation of Proximal Policy Optimization (PPO) Algorithm.

Author:     Anonymous Authors
"""
import torch
from sipga.algs.iwpg.iwpg import ImportanceWeightedPolicyGradientAlgorithm


class ProximalPolicyOptimizationAlgorithm(
    ImportanceWeightedPolicyGradientAlgorithm):
    def __init__(
            self,
            clip_ratio: float = 0.2,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.clip_ratio = clip_ratio

    def compute_loss_pi(self, data):
        # PPO Policy loss
        dist, _log_p = self.ac.pi(data['obs'], data['act'])
        ratio = torch.exp(_log_p - data['log_p'])

        clip_adv = data['adv'] * torch.clamp(
            ratio,
            1 - self.clip_ratio,
            1 + self.clip_ratio)
        loss_pi = -(torch.min(ratio * data['adv'], clip_adv)).mean()
        loss_pi -= self.entropy_coef * dist.entropy().mean()

        # Useful extra info
        approx_kl = (0.5 * (dist.mean - data['act']) ** 2
                     / dist.stddev ** 2).mean().item()
        ent = dist.entropy().mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, ratio=ratio.mean().item())

        return loss_pi, pi_info


def learn(
        env_id,
        **kwargs
) -> tuple:
    alg = ProximalPolicyOptimizationAlgorithm(
        env_id=env_id,
        **kwargs
    )

    ac, env = alg.learn()

    return ac, env
