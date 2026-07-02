"""DSRL-NA (noise-aliased DSRL) learner, following arXiv:2506.15799 and the
official implementation (ajwagen/dsrl, stable_baselines3/dsrl/dsrl.py), adapted
for a pi0-scale frozen policy.

Networks (on top of the PixelSACLearner's noise actor pi_W, noise critic Q_W and
temperature):
  - Q_A(s, a): action-space critic over the *denoised executed action chunks*,
    with a target network. Trained with a soft Bellman backup.
  - Q_W(s, w): noise-space critic (the parent's `_critic`). Trained purely by
    distillation from Q_A; the actor maximizes Q_W only.

Faithful-to-official parts:
  - buffer stores denoised executed chunks; Q_A is trained on them, so any
    action-space data trains Q_A regardless of which noise produced it
    ("noise aliasing").
  - Q_W_i regresses onto Q_A_i (ensemble-index-matched distillation).
  - actor maximizes the noise critic (never Q_A); no gradient flows through the
    frozen diffusion/flow policy.
  - soft (entropy-regularized) target in the Q_A backup, shared temperature.

Adaptations for a pi0-scale frozen policy (the official code denoises f(s, w)
on every training minibatch, which is intractable for pi0; the paper itself
used DSRL-SAC, not NA, for its pi0 experiments):
  - Q_W distillation uses the (s, w, a) aliasing pairs already generated during
    data collection (both w and a = f(s, w) are stored) instead of freshly
    sampled N(0, I) noise denoised at train time.
  - the Q_A Bellman target evaluates the next state via the distilled noise
    critic, y = r + gamma * mask * (Q_W_target(s', w') - alpha * log pi(w'|s'))
    with w' ~ pi_W(.|s'), instead of Q_A_target(s', f(s', w')). Q_W is the
    distilled estimate of Q_A(s, f(s, .)), so this replaces the denoising call
    with its learned surrogate; the target lag lives in Q_W_target.
"""
import copy
import functools
from typing import Dict, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.core.frozen_dict import FrozenDict
from flax.training import checkpoints
import pathlib

from jaxrl2.agents.pixel_sac.pixel_sac_learner import PixelSACLearner, TrainState
from jaxrl2.agents.pixel_sac.actor_updater import update_actor
from jaxrl2.agents.pixel_sac.temperature_updater import update_temperature
from jaxrl2.data.augmentations import batched_random_crop, color_transform
from jaxrl2.data.dataset import DatasetDict
from jaxrl2.networks.encoders.networks import Encoder, PixelMultiplexer, FeatureMultiplexer
from jaxrl2.networks.encoders.impala_encoder import ImpalaEncoder, SmallerImpalaEncoder
from jaxrl2.networks.encoders.resnet_encoderv1 import ResNet18, ResNet34, ResNetSmall
from jaxrl2.networks.encoders.resnet_encoderv2 import ResNetV2Encoder
from jaxrl2.networks.values import StateActionEnsemble
from jaxrl2.types import Params, PRNGKey
from jaxrl2.utils.target_update import soft_target_update


def update_qa_critic(key: PRNGKey, actor: TrainState, qa_critic: TrainState,
                     qw_target: TrainState,
                     temp: TrainState, batch: DatasetDict,
                     critic_reduction: str) -> Tuple[TrainState, Dict[str, float]]:
    """Soft Bellman update of the action-space critic Q_A on stored denoised chunks."""
    dist = actor.apply_fn({'params': actor.params}, batch['next_observations'])
    next_noise, next_log_probs = dist.sample_and_log_prob(seed=key)
    next_qs = qw_target.apply_fn({'params': qw_target.params},
                                 batch['next_observations'], next_noise)
    if critic_reduction == 'min':
        next_q = next_qs.min(axis=0)
    elif critic_reduction == 'mean':
        next_q = next_qs.mean(axis=0)
    else:
        raise NotImplementedError()

    alpha = temp.apply_fn({'params': temp.params})
    target_q = batch['rewards'] + batch['discount'] * batch['masks'] * (
        next_q - alpha * next_log_probs)
    target_q = jax.lax.stop_gradient(target_q)

    def qa_loss_fn(params: Params) -> Tuple[jnp.ndarray, Dict[str, float]]:
        qs = qa_critic.apply_fn({'params': params}, batch['observations'],
                                batch['env_actions'])
        loss = ((qs - target_q) ** 2).mean()
        return loss, {
            'qa_critic_loss': loss,
            'qa_q': qs.mean(),
            'qa_target_q': target_q.mean(),
            'qa_next_q_w': next_q.mean(),
        }

    grads, info = jax.grad(qa_loss_fn, has_aux=True)(qa_critic.params)
    new_qa_critic = qa_critic.apply_gradients(grads=grads)
    return new_qa_critic, info


def update_qw_distill(qw_critic: TrainState, qa_critic: TrainState,
                      batch: DatasetDict) -> Tuple[TrainState, Dict[str, float]]:
    """Distill Q_A into the noise critic on stored (s, w, a=f(s,w)) aliasing pairs.

    Index-matched: the i-th noise critic regresses onto the i-th action critic,
    as in the official update_noise_critic.
    """
    qa_vals = qa_critic.apply_fn({'params': qa_critic.params},
                                 batch['observations'], batch['env_actions'])
    qa_vals = jax.lax.stop_gradient(qa_vals)

    def qw_loss_fn(params: Params) -> Tuple[jnp.ndarray, Dict[str, float]]:
        qw = qw_critic.apply_fn({'params': params}, batch['observations'],
                                batch['actions'])
        loss = ((qw - qa_vals) ** 2).mean()
        return loss, {
            'qw_distill_loss': loss,
            'qw_q': qw.mean(),
            'qw_distill_target': qa_vals.mean(),
        }

    grads, info = jax.grad(qw_loss_fn, has_aux=True)(qw_critic.params)
    new_qw_critic = qw_critic.apply_gradients(grads=grads)
    return new_qw_critic, info


@functools.partial(jax.jit, static_argnames=('critic_reduction', 'color_jitter', 'aug_next', 'num_cameras'))
def _update_na_jit(
    rng: PRNGKey, actor: TrainState,
    qw_critic: TrainState, qw_target_params: Params,
    qa_critic: TrainState, qa_target_params: Params,
    temp: TrainState, batch: TrainState,
    tau: float, target_entropy: float,
    critic_reduction: str, color_jitter: bool, aug_next: bool, num_cameras: int,
) -> Tuple[PRNGKey, TrainState, TrainState, Params, TrainState, Params, TrainState, Dict[str, float]]:
    has_pixels = 'pixels' in batch['observations']
    if has_pixels:
        aug_pixels = batch['observations']['pixels']
        if batch['observations']['pixels'].squeeze().ndim != 2:
            rng, key = jax.random.split(rng)
            aug_pixels = batched_random_crop(key, batch['observations']['pixels'])
            if color_jitter:
                rng, key = jax.random.split(rng)
                if num_cameras > 1:
                    for i in range(num_cameras):
                        aug_pixels = aug_pixels.at[:, :, :, i*3:(i+1)*3].set((color_transform(key, aug_pixels[:, :, :, i*3:(i+1)*3].astype(jnp.float32)/255.)*255).astype(jnp.uint8))
                else:
                    aug_pixels = (color_transform(key, aug_pixels.astype(jnp.float32)/255.)*255).astype(jnp.uint8)
        observations = batch['observations'].copy(add_or_replace={'pixels': aug_pixels})
        batch = batch.copy(add_or_replace={'observations': observations})

        if aug_next:
            rng, key = jax.random.split(rng)
            aug_next_pixels = batched_random_crop(key, batch['next_observations']['pixels'])
            if color_jitter:
                rng, key = jax.random.split(rng)
                if num_cameras > 1:
                    for i in range(num_cameras):
                        aug_next_pixels = aug_next_pixels.at[:, :, :, i*3:(i+1)*3].set((color_transform(key, aug_next_pixels[:, :, :, i*3:(i+1)*3].astype(jnp.float32)/255.)*255).astype(jnp.uint8))
                else:
                    aug_next_pixels = (color_transform(key, aug_next_pixels.astype(jnp.float32)/255.)*255).astype(jnp.uint8)
            next_observations = batch['next_observations'].copy(
                add_or_replace={'pixels': aug_next_pixels})
            batch = batch.copy(add_or_replace={'next_observations': next_observations})

    # 1) Q_A: soft Bellman backup through the distilled noise critic's target
    key, rng = jax.random.split(rng)
    qw_target = qw_critic.replace(params=qw_target_params)
    new_qa_critic, qa_info = update_qa_critic(key, actor, qa_critic, qw_target,
                                              temp, batch, critic_reduction)
    # Q_A's target copy is intentionally never evaluated: the backup lag lives
    # in Q_W_target (see module docstring). Kept polyak-updated only so a
    # future f(s,w)-surrogate backup could switch to it without retraining.
    new_qa_target_params = soft_target_update(new_qa_critic.params, qa_target_params, tau)

    # 2) Q_W: ensemble-index-matched distillation from the fresh Q_A
    new_qw_critic, qw_info = update_qw_distill(qw_critic, new_qa_critic, batch)
    new_qw_target_params = soft_target_update(new_qw_critic.params, qw_target_params, tau)

    # 3) actor maximizes the noise critic (entropy-regularized), 4) temperature
    key, rng = jax.random.split(rng)
    new_actor, actor_info = update_actor(key, actor, new_qw_critic, temp, batch,
                                         critic_reduction=critic_reduction)
    new_temp, alpha_info = update_temperature(temp, actor_info['entropy'], target_entropy)

    return rng, new_actor, new_qw_critic, new_qw_target_params, new_qa_critic, new_qa_target_params, new_temp, {
        **qa_info,
        **qw_info,
        **actor_info,
        **alpha_info,
    }


class PixelSACNALearner(PixelSACLearner):

    def __init__(self,
                 seed: int,
                 observations: Union[jnp.ndarray, DatasetDict],
                 actions: jnp.ndarray,
                 env_actions: jnp.ndarray,
                 qa_critic_lr: float = 3e-4,
                 **kwargs):
        # parent builds: noise actor, noise critic Q_W (+ target), temperature
        super().__init__(seed, observations, actions, **kwargs)

        hidden_dims = kwargs.get('hidden_dims', (256, 256))
        if len(hidden_dims) == 1:
            hidden_dims = (hidden_dims[0], hidden_dims[0], hidden_dims[0])
        num_qs = kwargs.get('num_qs', 2)
        latent_dim = kwargs.get('latent_dim', 50)
        use_bottleneck = kwargs.get('use_bottleneck', True)
        obs_mode = kwargs.get('obs_mode', 'pixels')
        encoder_type = kwargs.get('encoder_type', 'resnet_34_v1')
        encoder_norm = kwargs.get('encoder_norm', 'group')
        use_spatial_softmax = kwargs.get('use_spatial_softmax', True)
        softmax_temperature = kwargs.get('softmax_temperature', 1)
        cnn_features = kwargs.get('cnn_features', (32, 32, 32, 32))
        cnn_strides = kwargs.get('cnn_strides', (2, 1, 1, 1))
        cnn_padding = kwargs.get('cnn_padding', 'VALID')

        if obs_mode == 'vlm':
            encoder_def = None
        elif encoder_type == 'small':
            encoder_def = Encoder(cnn_features, cnn_strides, cnn_padding)
        elif encoder_type == 'impala':
            encoder_def = ImpalaEncoder()
        elif encoder_type == 'impala_small':
            encoder_def = SmallerImpalaEncoder()
        elif encoder_type == 'resnet_small':
            encoder_def = ResNetSmall(norm=encoder_norm, use_spatial_softmax=use_spatial_softmax, softmax_temperature=softmax_temperature)
        elif encoder_type == 'resnet_18_v1':
            encoder_def = ResNet18(norm=encoder_norm, use_spatial_softmax=use_spatial_softmax, softmax_temperature=softmax_temperature)
        elif encoder_type == 'resnet_34_v1':
            encoder_def = ResNet34(norm=encoder_norm, use_spatial_softmax=use_spatial_softmax, softmax_temperature=softmax_temperature)
        elif encoder_type == 'resnet_small_v2':
            encoder_def = ResNetV2Encoder(stage_sizes=(1, 1, 1, 1), norm=encoder_norm)
        elif encoder_type == 'resnet_18_v2':
            encoder_def = ResNetV2Encoder(stage_sizes=(2, 2, 2, 2), norm=encoder_norm)
        elif encoder_type == 'resnet_34_v2':
            encoder_def = ResNetV2Encoder(stage_sizes=(3, 4, 6, 3), norm=encoder_norm)
        else:
            raise ValueError('encoder type not found!')

        self._rng, qa_key = jax.random.split(self._rng)

        qa_def = StateActionEnsemble(hidden_dims, num_qs=num_qs)
        if obs_mode == 'vlm':
            qa_def = FeatureMultiplexer(network=qa_def,
                                        latent_dim=latent_dim,
                                        use_bottleneck=use_bottleneck)
        else:
            qa_def = PixelMultiplexer(encoder=encoder_def,
                                      network=qa_def,
                                      latent_dim=latent_dim,
                                      use_bottleneck=use_bottleneck)
        qa_init = qa_def.init(qa_key, observations, env_actions)
        qa_params = qa_init['params']
        qa_batch_stats = qa_init['batch_stats'] if 'batch_stats' in qa_init else None
        self._qa_critic = TrainState.create(apply_fn=qa_def.apply,
                                            params=qa_params,
                                            tx=optax.adam(learning_rate=qa_critic_lr),
                                            batch_stats=qa_batch_stats)
        self._qa_target_critic_params = copy.deepcopy(qa_params)
        print('DSRL-NA: added action-space critic Q_A, env action sample shape',
              np.asarray(env_actions).shape)

    def update(self, batch: FrozenDict) -> Dict[str, float]:
        (new_rng, new_actor, new_qw_critic, new_qw_target_params,
         new_qa_critic, new_qa_target_params, new_temp, info) = _update_na_jit(
            self._rng, self._actor,
            self._critic, self._target_critic_params,
            self._qa_critic, self._qa_target_critic_params,
            self._temp, batch,
            self.tau, self.target_entropy,
            self.critic_reduction, self.color_jitter, self.aug_next, self.num_cameras)

        self._rng = new_rng
        self._actor = new_actor
        self._critic = new_qw_critic
        self._target_critic_params = new_qw_target_params
        self._qa_critic = new_qa_critic
        self._qa_target_critic_params = new_qa_target_params
        self._temp = new_temp
        return info

    @property
    def _save_dict(self):
        save_dict = {
            'critic': self._critic,
            'target_critic_params': self._target_critic_params,
            'qa_critic': self._qa_critic,
            'qa_target_critic_params': self._qa_target_critic_params,
            'actor': self._actor,
            'temp': self._temp
        }
        return save_dict

    def restore_checkpoint(self, dir):
        assert pathlib.Path(dir).exists(), f"Checkpoint {dir} does not exist."
        output_dict = checkpoints.restore_checkpoint(dir, self._save_dict)
        self._actor = output_dict['actor']
        self._critic = output_dict['critic']
        self._target_critic_params = output_dict['target_critic_params']
        self._qa_critic = output_dict['qa_critic']
        self._qa_target_critic_params = output_dict['qa_target_critic_params']
        self._temp = output_dict['temp']
        print('restored from ', dir)
