import torch

import numpy as np
from torch import nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Union, Optional

from tianshou.policy import PGPolicy
from tianshou.data import Batch, ReplayBuffer
from tianshou.data import to_torch
from numba import njit
import sys

sys.path.append("..")
from util import to_numpy, to_torch_as


def _episodic_return(
    v_s_: np.ndarray, rew: np.ndarray, done: np.ndarray, gamma: float, gae_lambda: float,
) -> np.ndarray:
    """Numba speedup: 4.1s -> 0.057s."""
    returns = np.roll(v_s_, 1)
    m = (1.0 - done) * gamma
    delta = rew + v_s_ * m - returns
    m *= gae_lambda
    gae = 0.0
    for i in range(len(rew) - 1, -1, -1):
        gae_new = delta[i] + m[i] * gae
        gae = gae_new
        returns[i] += gae
    return returns


class PPOwoAC(PGPolicy):
    """ The PPO policy with Teacher supervision"""

    def __init__(
        self,
        actor: torch.nn.Module,
        critic: torch.nn.Module,
        optim: torch.optim.Optimizer,
        dist_fn: torch.distributions.Distribution,
        teacher=None,
        discount_factor: float = 0.99,
        max_grad_norm: Optional[float] = None,
        eps_clip: float = 0.2,
        vf_clip_para=10.0,
        vf_coef: float = 0.5,
        kl_coef=0.5,
        kl_target=0.01,
        ent_coef: float = 0.01,
        sup_coef=0.1,
        action_range: Optional[Tuple[float, float]] = None,
        gae_lambda: float = 0.95,
        dual_clip: Optional[float] = None,
        value_clip: bool = True,
        reward_normalization: bool = True,
        **kwargs
    ) -> None:
        super().__init__(None, None, dist_fn, discount_factor, **kwargs)
        self._max_grad_norm = max_grad_norm
        self._eps_clip = eps_clip
        self._vf_clip_para = vf_clip_para
        self._w_vf = vf_coef
        self._w_ent = ent_coef
        self._range = action_range
        self.actor = actor
        self.critic = critic
        self.optim = optim
        self.sup_coef = sup_coef
        self.kl_target = kl_target
        self.kl_coef = kl_coef
        self._batch = 64
        assert 0 <= gae_lambda <= 1, "GAE lambda should be in [0, 1]."
        self._lambda = gae_lambda
        assert dual_clip is None or dual_clip > 1, "Dual-clip PPO parameter should greater than 1."
        self._dual_clip = dual_clip
        self._value_clip = value_clip
        self._rew_norm = reward_normalization
        if not teacher is None:
            self.teacher = torch.load(teacher, map_location=torch.device("cpu"))
            self.teacher.to(self.actor.device)
            self.teacher.actor.extractor.device = self.actor.device
        else:
            self.teacher = None

    @staticmethod
    def compute_episodic_return(
        batch: Batch,
        v_s_: Optional[Union[np.ndarray, torch.Tensor]] = None,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        rew_norm: bool = False,
    ) -> Batch:
        """Compute returns over given full-length episodes.
        Implementation of Generalized Advantage Estimator (arXiv:1506.02438).
        :param batch: a data batch which contains several full-episode data
            chronologically.
        :type batch: :class:`~tianshou.data.Batch`
        :param v_s_: the value function of all next states :math:`V(s')`.
        :type v_s_: numpy.ndarray
        :param float gamma: the discount factor, should be in [0, 1], defaults
            to 0.99.
        :param float gae_lambda: the parameter for Generalized Advantage
            Estimation, should be in [0, 1], defaults to 0.95.
        :param bool rew_norm: normalize the reward to Normal(0, 1), defaults
            to False.
        :return: a Batch. The result will be stored in batch.returns as a numpy
            array with shape (bsz, ).
        """
        rew = batch.rew
        v_s_ = np.zeros_like(rew) if v_s_ is None else to_numpy(v_s_.flatten())
        assert not np.isnan(v_s_).any()
        assert not np.isnan(rew).any()
        assert not np.isnan(batch.done).any()
        returns = _episodic_return(v_s_, rew, batch.done, gamma, gae_lambda)
        assert not np.isnan(returns).any()
        if rew_norm and not np.isclose(returns.std(), 0.0, 1e-2):
            returns = (returns - returns.mean()) / returns.std()
        assert not np.isnan(returns).any()
        batch.returns = returns
        return batch

    def process_fn(self, batch: Batch, buffer: ReplayBuffer, indice: np.ndarray) -> Batch:
        if self._rew_norm:
            mean, std = batch.rew.mean(), batch.rew.std()
            if not np.isclose(std, 0):
                batch.rew = (batch.rew - mean) / std
        assert not np.isnan(batch.rew).any()
        if self._lambda in [0, 1]:
            return self.compute_episodic_return(batch, None, gamma=self._gamma, gae_lambda=self._lambda)
        else:
            v_ = []
            with torch.no_grad():
                for b in batch.split(self._batch, shuffle=False):
                    v_.append(self.critic(b.obs_next))
            v_ = to_numpy(torch.cat(v_, dim=0))
            assert not np.isnan(v_).any()
            return self.compute_episodic_return(batch, v_, gamma=self._gamma, gae_lambda=self._lambda)

    def forward(self, batch: Batch, state: Optional[Union[dict, Batch, np.ndarray]] = None, **kwargs) -> Batch:
        """Compute action over the given batch data."""
        logits, h = self.actor(batch.obs, state=state, info=batch.info)
        if isinstance(logits, tuple):
            dist = self.dist_fn(*logits)
        else:
            dist = self.dist_fn(logits)
        if self.training:
            try:
                act = dist.sample() # 根据概率进行多项式分布采样, replacement表示有无放回采样
            except:
                print(logits)
                act = dist.sample()
        else:
            act = torch.argmax(logits, dim=1)
        if self._range:
            act = act.clamp(self._range[0], self._range[1])
        return Batch(logits=logits, act=act, state=h, dist=dist)

    def learn(self, batch: Batch, batch_size: int, repeat: int, **kwargs) -> Dict[str, List[float]]:
        self._batch = batch_size
        losses, clip_losses, vf_losses, ent_losses = [], [], [], []
        if self.teacher is not None:
            supervision_losses = []
        feature = []
        torch.cuda.empty_cache()

        batch.act = torch.tensor(batch.act).float().cuda()
        batch.returns = torch.tensor(batch.returns).float().cuda()
        if self._rew_norm:
            mean, std = batch.returns.mean(), batch.returns.std()
            if not np.isclose(std.item(), 0):
                batch.returns = (batch.returns - mean) / std
        batch.adv = batch.returns
        if self._rew_norm:
            mean, std = batch.adv.mean(), batch.adv.std()
            if not np.isclose(std.item(), 0):
                batch.adv = (batch.adv - mean) / std
        torch.cuda.empty_cache()
        for _ in range(repeat):
            for b in batch.split(batch_size):
                dist = self(b).dist
                ratio = dist.log_prob(b.act)
                loss = (ratio * b.adv).mean()
                losses.append(loss.item())
                self.optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()), self._max_grad_norm,
                )
                self.optim.step()

        res = {
            "loss/total_loss": losses,
            "reward": batch.rew.tolist(),
            'returns': batch.returns.detach().cpu().numpy().tolist()
        }
        if not self.teacher is None:
            res["loss/supervision"] = supervision_losses
        return res
