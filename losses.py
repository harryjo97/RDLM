import torch
import torch.optim as optim
import numpy as np
from model import utils as mutils
import random
from functools import partial

_LOSSES = {}


def register_loss_fn(cls=None, *, name=None):
    """A decorator for registering loss classes."""
    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _LOSSES:
            raise ValueError(f'Already registered model with name: {local_name}')
        _LOSSES[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)

    
def get_loss_fn(name):
    return _LOSSES[name]


def get_optimizer(config, params):
    if config.optim.optimizer == 'Adam':
        optimizer = optim.Adam(
            params, lr=config.optim.lr, 
            betas=(config.optim.beta1, config.optim.beta2), 
            eps=config.optim.eps,
            weight_decay=config.optim.weight_decay
        )
    elif config.optim.optimizer == 'AdamW':
        optimizer = optim.AdamW(
            params, lr=config.optim.lr, 
            betas=(config.optim.beta1, config.optim.beta2), 
            eps=config.optim.eps,
            weight_decay=config.optim.weight_decay
        )
    else:
        raise NotImplementedError(
            f'Optimizer {config.optim.optimizer} not supported yet!'
        )

    return optimizer


def optimization_manager(config):
    """Returns an optimize_fn based on `config`."""
    def optimize_fn(optimizer, 
                    scaler, 
                    params, 
                    step, 
                    lr=config.optim.lr,
                    warmup=config.optim.warmup,
                    grad_clip=config.optim.grad_clip):
        """Optimizes with warmup and gradient clipping (disabled if negative)."""
        scaler.unscale_(optimizer)

        if warmup > 0:
            for g in optimizer.param_groups:
                g['lr'] = lr * np.minimum(step / warmup, 1.0)
        if grad_clip >= 0:
            torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)

        scaler.step(optimizer)
        scaler.update()

    return optimize_fn


def get_step_fn(loss_type, sde, train, optimize_fn, accum, **kwargs):
    loss_fn = get_loss_fn(loss_type)(sde, train, **kwargs)

    accum_iter = 0
    total_loss = 0

    def step_fn(state, batch, cond=None):
        nonlocal accum_iter 
        nonlocal total_loss

        model = state['model']

        if train:
            optimizer = state['optimizer']
            scaler = state['scaler']
            loss = loss_fn(model, batch, cond=cond).mean() / accum
            
            scaler.scale(loss).backward()

            accum_iter += 1
            total_loss += loss.detach()
            if accum_iter == accum:
                accum_iter = 0

                state['step'] += 1
                optimize_fn(optimizer, scaler, model.parameters(), step=state['step'])
                state['ema'].update(model.parameters())
                optimizer.zero_grad()
                
                loss = total_loss
                total_loss = 0
        else:
            with torch.no_grad():
                ema = state['ema']
                ema.store(model.parameters())
                ema.copy_to(model.parameters())
                loss = loss_fn(model, batch, cond=cond).mean()
                ema.restore(model.parameters())

        return loss

    return step_fn


# ELBO loss function
@register_loss_fn(name="elbo")
def elbo_loss_fn(sde, train, sampling_eps=1e-4, **kwargs):
    simul_steps = kwargs.get("simul_steps", 0)
    interpolant_fn = sde.interpolant if simul_steps == 0 else \
        partial(sde.interpolant_simul, simul_steps=simul_steps)

    def loss_fn(model, batch, cond=None):
        """
        Batch shape: [B, L, D]
        """
        drift_fn = mutils.get_drift_fn(model, sde, train=train, sampling=False)

        if sde.scheduler.weight_type == "default" or not train:
            t = (1 - sampling_eps) * torch.rand(batch.shape[0], device=batch.device)
        else:
            t = sde.scheduler.importance_weighted_time((batch.shape[0],), batch.device)

        prior_sample = sde.prior_sample(batch.shape, batch.device, t=t)
        interpolant = interpolant_fn(prior_sample, batch, t)

        drift = sde.drift(batch, interpolant, t)
        loss = sde.manifold.squared_norm(drift_fn(interpolant, t) - drift)
        loss = 0.5 * loss.sum(dim=-1) * sde.scheduler.importance_weight(t, train) / sde.diffusion(interpolant, t).square()

        return loss
     
    return loss_fn


# Cross-entropy loss function
@register_loss_fn(name="ce")
def ce_loss_fn(sde, train, sampling_eps=1e-4, **kwargs):
    simul_steps = kwargs.get("simul_steps", 0)
    interpolant_fn = sde.interpolant if simul_steps == 0 else \
        partial(sde.interpolant_simul, simul_steps=simul_steps)

    def loss_fn(model, batch, cond=None):
        """
        Batch shape: [B, L, D]
        """
        model_fn = mutils.get_model_fn(model, train=train)

        if sde.scheduler.weight_type == "default" or not train:
            t = (1 - sampling_eps) * torch.rand(batch.shape[0], device=batch.device)
        else:
            t = sde.scheduler.importance_weighted_time((batch.shape[0],), batch.device)

        prior_sample = sde.prior_sample(batch.shape, batch.device, t=t)
        interpolant = interpolant_fn(prior_sample, batch, t)

        output = model_fn(interpolant, t)
        loss = torch.vmap(torch.nn.CrossEntropyLoss(reduction='sum'))(
            output.to(torch.float32), batch[...,:output.shape[-1]].argmax(-1)
        )
        loss = loss * sde.scheduler.importance_weight(t, train)

        return loss
     
    return loss_fn