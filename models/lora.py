"""Frozen MLP + dense mask-gated LoRA atom banks, adapted from upstream lora_ftn.

All atoms remain on the search graph, including inactive ones. The same mask
selects atoms in every layer; the robotics action head applies tanh.
"""
import math
import torch
from torch import nn


class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, H, k, cfg):
        super().__init__()
        self.scale = cfg.lora_alpha / k
        self.base = nn.Linear(in_features, out_features, bias=False)
        with torch.no_grad():
            self.base.weight.normal_(0, cfg.lora_weight_scale / math.sqrt(in_features))
        self.base.requires_grad_(False)
        self.A = nn.Parameter(torch.empty(H, in_features, cfg.lora_rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.B = nn.Parameter(torch.empty(H, cfg.lora_rank, out_features).normal_(0, cfg.lora_b0_scale))
        self.bias_bank = (nn.Parameter(torch.empty(H, out_features).normal_(0, cfg.lora_b0_scale))
                          if cfg.lora_bias else None)
        self.dropout = nn.Dropout(cfg.lora_dropout)

    def forward(self, x, masks):
        # x: (S,L,I), masks: (S,H). Contract weights first to avoid materializing
        # (S,L,H,R) activations for large segment batches. The contraction is
        # mathematically identical to upstream's dense gated atom computation.
        weights = torch.einsum('sh,hir,hro->sio', masks, self.A, self.B)
        result = self.base(x) + self.scale * torch.bmm(self.dropout(x), weights)
        if self.bias_bank is not None:
            result = result + self.scale * (masks @ self.bias_bank).unsqueeze(1)
        return result


class LoRAActor(nn.Module):
    def __init__(self, obs_dim, act_dim, cfg):
        super().__init__()
        self.H = cfg.ftn_hidden_size
        sizes = [obs_dim] + [cfg.lora_width] * cfg.lora_layers + [act_dim]
        self.layers = nn.ModuleList([LoRALinear(a, b, self.H, cfg.ftn_k, cfg)
                                    for a, b in zip(sizes, sizes[1:])])

    def forward_batched(self, obs, masks):
        h = obs
        for index, layer in enumerate(self.layers):
            h = layer(h, masks)
            if index < len(self.layers) - 1:
                h = torch.relu(h)
        return torch.tanh(h)

    def forward(self, obs, mask):
        return self.forward_batched(obs.unsqueeze(0), mask.unsqueeze(0)).squeeze(0)

    def unit_parameters(self):
        """Only the atom banks are trainable; the shared backbone stays frozen."""
        for layer in self.layers:
            yield layer.A, 0
            yield layer.B, 0
            if layer.bias_bank is not None:
                yield layer.bias_bank, 0
