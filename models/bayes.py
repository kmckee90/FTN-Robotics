"""Vectorized upstream Bayesian mask inference and loss-based changepoints.

Each row is an independent posterior. This generalizes bayesian_adv's scalar
algorithm without combining KL terms, candidate selection, or detector history
between MT10 segments. Actor weights are frozen by the caller.
"""
import math
import torch
from torch import nn


def topk(scores, k):
    return torch.zeros_like(scores).scatter_(-1, scores.topk(k, dim=-1).indices, 1)


def sample_mask(logits, k, temperature, stochastic):
    if stochastic:
        u = torch.rand_like(logits).clamp_(1e-7, 1 - 1e-7)
        return topk(logits / temperature - torch.log(-torch.log(u)), k)
    return topk(logits, k)


def bernoulli_kl(logits, prior, temperature):
    eps = torch.finfo(logits.dtype).eps
    q = torch.sigmoid(logits / temperature).clamp(eps, 1 - eps)
    p = torch.sigmoid(prior / temperature).clamp(eps, 1 - eps)
    return (q * (q.log() - p.log()) + (1 - q) * ((1 - q).log() - (1 - p).log())).mean(-1)


def overlap_penalty(mask, logits, reference, cfg):
    if reference is None or cfg.reconfig_overlap_lambda == 0:
        return mask.new_zeros(mask.shape[0])
    if cfg.reconfig_overlap_mode == 'seen_logits':
        cost = (logits * reference).mean(-1)
    else:
        overlap = (mask * reference).sum(-1) / reference.sum(-1).clamp_min(1)
        cost = (overlap * (1 - overlap) if cfg.reconfig_overlap_mode == 'bump' else
                overlap * torch.sigmoid((cfg.reconfig_overlap_threshold - overlap)
                                        / cfg.reconfig_overlap_temperature))
    return cfg.reconfig_overlap_lambda * cost


@torch.enable_grad()
def posterior_update(loss_fn, previous, reference, cfg, changepoint=False, steps=None):
    prefix = 'bayes_changepoint_' if changepoint else 'bayes_'
    steps = getattr(cfg, prefix + 'reconfig_steps') if steps is None else steps
    lr = getattr(cfg, prefix + 'reconfig_lr')
    temperature = getattr(cfg, prefix + 'temperature')
    decay = getattr(cfg, prefix + 'prior_decay')
    stochastic = getattr(cfg, prefix + 'stochastic_mask')
    kl_weight = getattr(cfg, prefix + 'kl_weight')
    prior = (previous * decay).detach().clone()
    logits = nn.Parameter(prior.clone())
    optimizer = torch.optim.Adam([logits], lr=lr)
    lowest = previous.new_full((len(previous),), float('inf'))
    best_mask, best_logits = topk(prior, cfg.ftn_k), prior.clone()
    first = None
    for i in range(steps):
        if cfg.bayes_sample_type == 'annealste':
            soft = torch.sigmoid(logits)
            tau = logits.detach().std(-1, keepdim=True, unbiased=logits.shape[-1] > 1) * math.exp(-i / max(cfg.reconfig_steps / 4, 1e-8))
            noise = temperature * torch.randn_like(logits) * tau * cfg.reconfig_noise_scale
            hard = topk(logits.detach() + noise, cfg.ftn_k)
        else:
            soft = torch.sigmoid(logits / temperature)
            anneal = math.exp(-i / max((steps - 1) / 3, 10))
            hard = sample_mask(logits.detach(), cfg.ftn_k, temperature * anneal, stochastic)
        mask = hard + soft - soft.detach()
        task = loss_fn(mask)
        if first is None:
            first = task.detach()
        kl = bernoulli_kl(logits, prior, temperature)
        penalty = overlap_penalty(mask, logits, reference, cfg)
        objective = task + kl_weight * kl + penalty
        with torch.no_grad():
            selection = task + penalty  # upstream best selection excludes KL
            improved = selection < lowest
            lowest = torch.minimum(selection, lowest)
            best_mask = torch.where(improved[:, None], hard, best_mask)
            best_logits = torch.where(improved[:, None], logits.detach(), best_logits)
        optimizer.zero_grad()
        objective.sum().backward()
        optimizer.step()
    if cfg.bayes_reconfig_take_best and steps:
        result, updated = best_mask, best_logits
    else:
        updated = logits.detach().clone()
        result = sample_mask(updated, cfg.ftn_k, temperature, stochastic)
    with torch.no_grad():
        final = loss_fn(result)
        kl = bernoulli_kl(updated, prior, temperature)
        penalty = overlap_penalty(result, updated, reference, cfg)
    return result, updated, {'bc_init': final if first is None else first, 'bc_final': final,
                             'kl': kl, 'objective': final + kl_weight * kl + penalty,
                             'overlap_penalty': penalty, 'changepoint': changepoint,
                             'prior_decay': decay, 'steps': steps}


@torch.enable_grad()
def cold_search(loss_fn, template, cfg, steps):
    """Fresh zero-logit inference: no training posterior, KL or overlap reference."""
    best = topk(torch.zeros_like(template), cfg.ftn_k)
    best_logits = torch.zeros_like(template)
    lowest = template.new_full((len(template),), float('inf'))
    first = None
    for _ in range(cfg.eval_reconfig_restarts):
        logits = nn.Parameter(torch.zeros_like(template))
        opt = torch.optim.Adam([logits], lr=cfg.eval_reconfig_lr)
        for i in range(steps):
            soft = torch.sigmoid(logits / cfg.eval_reconfig_temperature)
            anneal = math.exp(-i / max((steps - 1) / 3, 10))
            hard = sample_mask(logits.detach(), cfg.ftn_k, cfg.eval_reconfig_temperature * anneal, True)
            loss = loss_fn(hard + soft - soft.detach())
            if first is None:
                first = loss.detach()
            if cfg.eval_reconfig_take_best:
                improved = loss.detach() < lowest
                best = torch.where(improved[:, None], hard, best)
                best_logits = torch.where(improved[:, None], logits.detach(), best_logits)
                lowest = torch.minimum(lowest, loss.detach())
            opt.zero_grad()
            loss.sum().backward()
            opt.step()
        candidate = topk(logits.detach(), cfg.ftn_k)
        with torch.no_grad():
            loss = loss_fn(candidate)
            improved = loss < lowest
            best = torch.where(improved[:, None], candidate, best)
            best_logits = torch.where(improved[:, None], logits.detach(), best_logits)
            lowest = torch.minimum(lowest, loss)
    return best, best_logits, {'bc_init': lowest if first is None else first, 'bc_final': lowest}


class InferenceState(nn.Module):
    """Non-parameter state, banked by identity (one stream or stable segment IDs)."""
    def __init__(self, rows, H, device=None):
        super().__init__()
        self.register_buffer('logits', torch.zeros(rows, H, device=device))
        self.register_buffer('masks', torch.zeros(rows, H, device=device))
        self.register_buffer('mean', torch.zeros(rows, device=device))
        self.register_buffer('variance', torch.zeros(rows, device=device))
        self.register_buffer('since', torch.zeros(rows, dtype=torch.long, device=device))
        self.register_buffer('initialized', torch.zeros(rows, dtype=torch.bool, device=device))

    def select(self, ids, device):
        ids = torch.as_tensor(ids, dtype=torch.long, device=self.logits.device)
        return {key: value[ids].to(device) for key, value in self.named_buffers()}

    @torch.no_grad()
    def put(self, ids, values):
        ids = torch.as_tensor(ids, dtype=torch.long, device=self.logits.device)
        for key, value in values.items():
            buffer = getattr(self, key)
            buffer[ids] = value.detach().to(buffer.device)


def detect_changepoints(losses, state, cfg):
    warmup = math.ceil(1 / max(1 - cfg.reconfig_reactive_ema, 1e-8))
    return ~state['initialized'] | ((state['since'] >= warmup) &
        (losses > state['mean'] + cfg.reconfig_reactive_sd * state['variance'].clamp_min(0).sqrt()))


def update_detector(state, post_losses, changed, cfg):
    alpha = cfg.reconfig_reactive_ema
    diff = post_losses - state['mean']
    state['variance'] = torch.where(changed, torch.zeros_like(diff), alpha * state['variance'] + (1-alpha) * diff.square())
    state['mean'] = torch.where(changed, post_losses, alpha * state['mean'] + (1-alpha) * post_losses)
    state['since'] = torch.where(changed, torch.zeros_like(state['since']), state['since']) + 1
    state['initialized'] = torch.ones_like(state['initialized'])
