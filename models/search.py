"""Shared search plumbing for scalar task and parallel segment inference."""
from contextlib import contextmanager
import math
import torch
from torch import nn
from .bayes import (InferenceState, cold_search, detect_changepoints, overlap_penalty,
                    posterior_update, topk, update_detector)
from .config import is_bayesian, model_name, reactive_enabled


@contextmanager
def frozen_actor(actor):
    flags = [p.requires_grad for p in actor.parameters()]
    training = actor.training
    try:
        actor.eval()
        actor.requires_grad_(False)
        yield
    finally:
        actor.train(training)
        for p, flag in zip(actor.parameters(), flags):
            p.requires_grad_(flag)


@torch.enable_grad()
def lora_search(loss_fn, template, reference, cfg, steps, init=None, stored=None):
    logits = nn.Parameter((cfg.mask_init_noise_scale * torch.randn_like(template)) if init is None else init.detach().clone())
    opt = getattr(torch.optim, cfg.lora_reconfig_optimizer)([logits], lr=cfg.reconfig_lr)
    lowest = template.new_full((len(template),), float('inf'))
    best, best_logits = topk(logits.detach(), cfg.ftn_k), logits.detach().clone()
    first = None
    matched = False
    do_match = cfg.lora_match_stored and cfg.lora_reconfig_algorithm == 'annealste'
    def match():
        nonlocal best, lowest
        if stored is None or not len(stored):
            return
        # Nearest stored masks by overlap, then choose by support loss per row.
        indices = (best @ stored.T).topk(min(cfg.lora_match_stored_topk, len(stored)), dim=-1).indices
        candidates = stored[indices]
        matched_loss = torch.full_like(lowest, float('inf'))
        matched_mask = best.clone()
        for i in range(candidates.shape[1]):
            candidate = candidates[:, i]
            score = loss_fn(candidate) + overlap_penalty(candidate, logits, reference, cfg)
            improve = score < matched_loss
            matched_mask = torch.where(improve[:, None], candidate, matched_mask)
            matched_loss = torch.minimum(matched_loss, score)
        best, lowest = matched_mask, matched_loss
    for i in range(steps):
        if cfg.lora_reconfig_algorithm == 'annealste':
            tau = logits.detach().std(-1, keepdim=True, unbiased=logits.shape[-1] > 1) * math.exp(-i / max(1, steps / 4))
            hard = topk(logits.detach() + torch.randn_like(logits) * tau * cfg.reconfig_noise_scale, cfg.ftn_k)
            mask = hard + logits - logits.detach()
        else:
            mask = torch.sigmoid(logits)
            hard = topk(logits.detach(), cfg.ftn_k)
        task = loss_fn(mask)
        if first is None:
            first = task.detach()
        loss = task + overlap_penalty(mask, logits, reference, cfg)
        with torch.no_grad():
            # Score the actual hard candidate, including for the soft fast path.
            score = loss_fn(hard) + overlap_penalty(hard, logits, reference, cfg)
            improve = score < lowest
            best = torch.where(improve[:, None], hard, best)
            best_logits = torch.where(improve[:, None], logits.detach(), best_logits)
            lowest = torch.minimum(lowest, score)
        opt.zero_grad()
        loss.sum().backward()
        opt.step()
        if do_match and not matched and i + 1 >= cfg.lora_match_stored_after:
            with torch.no_grad():
                match()
            matched = True
    with torch.no_grad():
        candidate = topk(logits.detach(), cfg.ftn_k)
        score = loss_fn(candidate) + overlap_penalty(candidate, logits, reference, cfg)
        improve = score < lowest
        best = torch.where(improve[:, None], candidate, best)
        best_logits = torch.where(improve[:, None], logits.detach(), best_logits)
        lowest = torch.minimum(lowest, score)
        if do_match and not matched:
            match()
        final = loss_fn(best)
    return best, best_logits, {'bc_init': final if first is None else first, 'bc_final': final}


class ModelInference:
    def __init__(self, actor, cfg, smoother):
        self.actor, self.cfg, self.smoother = actor, cfg, smoother
        self.state = InferenceState(1, actor.H, device=next(actor.parameters()).device)
        self.segment_state = None
        self.seen = torch.zeros(actor.H, device=next(actor.parameters()).device)
        self.last_stats = {}

    def init_segments(self, count):
        # Potentially many segments: bank on CPU, transfer only the active rows.
        self.segment_state = InferenceState(count, self.actor.H, device='cpu')

    def record_masks(self, masks):
        with torch.no_grad():
            self.seen.copy_(torch.maximum(self.seen, masks.reshape(-1, self.actor.H).amax(0)))

    def state_dict(self):
        return {'stream': {k: v.detach().cpu() for k, v in self.state.state_dict().items()},
                'segments': None if self.segment_state is None else self.segment_state.state_dict(),
                'seen': self.seen.detach().cpu()}

    def load_state_dict(self, saved):
        self.state.load_state_dict(saved['stream'])
        if saved.get('segments') is not None:
            self.init_segments(len(saved['segments']['logits']))
            self.segment_state.load_state_dict(saved['segments'])
        self.seen.copy_(saved.get('seen', torch.zeros_like(self.seen)))

    @torch.enable_grad()
    def search(self, obs, actions, commit, steps=None, ids=None, init_logits=None, stored=None):
        cfg = self.cfg
        bank = self.state if ids is None else self.segment_state
        if bank is None:
            raise ValueError('Initialize segment inference state before selecting segment IDs')
        ids = [0] if ids is None else ids
        state = bank.select(ids, obs.device)
        if len(state['logits']) != len(obs):
            raise ValueError('Each support batch needs its own inference-state row')
        template = state['logits']
        if cfg.lora_match_stored and self.segment_state is not None and stored is None:
            known = self.segment_state.initialized
            stored = self.segment_state.masks[known].to(obs.device)
        def loss_for(o, a):
            return lambda masks: (self.actor.forward_batched(o, masks) - a).square().mean((1, 2))
        with frozen_actor(self.actor):
            if not commit:
                n = cfg.eval_reconfig_sample_size
                loss_fn = loss_for(obs[:, :n], actions[:, :n])
                budget = cfg.reconfig_steps if steps is None else steps
                masks, logits, stats = cold_search(loss_fn, template, cfg, budget)
            else:
                before = loss_for(obs, actions)(state['masks']).detach()
                reactive = reactive_enabled(cfg)
                changed = detect_changepoints(before, state, cfg) if reactive else torch.zeros_like(state['initialized'])
                masks, logits = state['masks'].clone(), template.clone()
                first, final = before.clone(), before.clone()
                kl = before.new_zeros(len(before))
                budgets = torch.zeros_like(state['since'])
                for change in (False, True):
                    selected = (changed == change).nonzero().flatten()
                    if not len(selected):
                        continue
                    if reactive and not change and not is_bayesian(cfg) and not cfg.lora_reactive_continuous_updates:
                        continue
                    sample_size = (cfg.bayes_changepoint_reconfig_sample_size if change else cfg.bayes_reconfig_sample_size) if is_bayesian(cfg) else None
                    o, a = obs[selected, :sample_size], actions[selected, :sample_size]
                    loss_fn = loss_for(o, a)
                    reference = (self.seen.expand(len(selected), -1) if cfg.reconfig_overlap_mode == 'seen_logits' else state['masks'][selected])
                    if is_bayesian(cfg):
                        m, l, info = posterior_update(loss_fn, template[selected], reference, cfg, change, steps)
                        kl[selected] = info['kl']
                        budgets[selected] = info['steps']
                    else:
                        budget = (cfg.lora_changepoint_reconfig_steps
                                  if change and model_name(cfg) == 'lora' else cfg.reconfig_steps)
                        budget = budget if steps is None else steps
                        initial = None
                        if not change and (cfg.lora_warm_start_reconfig or cfg.mask_warm_start):
                            initial = template[selected] if init_logits is None else init_logits[selected]
                            if not state['initialized'][selected].any() and init_logits is None:
                                initial = None
                        m, l, info = lora_search(loss_fn, template[selected], reference, cfg, budget, initial, stored)
                        budgets[selected] = budget
                    masks[selected], logits[selected] = m, l
                    first[selected], final[selected] = info['bc_init'], info['bc_final']
                stats = {'bc_init': first, 'bc_final': final, 'kl': kl, 'changepoint': changed, 'steps': budgets}
            if self.smoother.enabled:
                masks = torch.stack([self.smoother.smooth(m) for m in masks])
            with torch.no_grad():
                stats['bc_final'] = loss_for(obs, actions)(masks)
            if commit:
                state['logits'], state['masks'] = logits.detach(), masks.detach()
                update_detector(state, stats['bc_final'], changed, cfg)
                bank.put(ids, state)
                self.last_stats = {k: v.detach().clone() if torch.is_tensor(v) else v for k, v in stats.items()}
        return masks.detach(), logits.detach(), stats
