"""A tiny causal transformer for synthetic key/value recall."""

from __future__ import annotations

import hashlib
import random

import torch
from torch import nn

BOS, QUERY = 0, 1
KEYS, VALUES = list(range(2, 10)), list(range(10, 18))
VOCAB = ["·", "?"] + list("ABCDEFGH") + [str(i) for i in range(8)]


def sample(rng, pairs=4):
    keys = rng.sample(KEYS, pairs)
    values = [rng.randrange(8) for _ in keys]
    q = rng.randrange(pairs)
    tokens = [BOS]
    for k, v in zip(keys, values):
        tokens.extend([k, v + 10])
    return tokens + [QUERY, keys[q]], values[q]


def identifier(tokens):
    return hashlib.sha256(bytes(tokens)).hexdigest()


def dataset(n, seed, split, pairs=4):
    if split not in {"train", "dev", "test"}:
        raise ValueError("split must be train, dev or test")
    rng, seen, rows = random.Random(seed), set(), []
    while len(rows) < n:
        x, y = sample(rng, pairs)
        key = identifier(x)
        bucket = int(key[:8], 16) % 10
        matches = bucket < 8 if split == "train" else bucket == (8 if split == "dev" else 9)
        if key not in seen and matches:
            rows.append((x, y))
            seen.add(key)
    return rows


class Attention(nn.Module):
    def __init__(self, width=64, heads=4):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(width, width * 3)
        self.out = nn.Linear(width, width)

    def forward(self, x, zero_head=None):
        b, t, d = x.shape
        qkv = self.qkv(x).view(b, t, 3, self.heads, d // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        logits = q @ k.transpose(-1, -2) / (d // self.heads) ** 0.5
        mask = torch.ones(t, t, device=x.device, dtype=torch.bool).triu(1)
        weights = logits.masked_fill(mask, float("-inf")).softmax(-1)
        context = weights @ v
        if zero_head is not None:
            context = context.clone()
            context[:, zero_head] = 0
        return self.out(context.transpose(1, 2).reshape(b, t, d)), weights


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(64), nn.LayerNorm(64)
        self.attention = Attention()
        self.mlp = nn.Sequential(nn.Linear(64, 128), nn.GELU(), nn.Linear(128, 64))

    def forward(self, x, zero_head=None):
        update, attention = self.attention(self.norm1(x), zero_head)
        x = x + update
        return x + self.mlp(self.norm2(x)), attention


class RecallTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.token = nn.Embedding(18, 64)
        self.position = nn.Embedding(17, 64)
        self.blocks = nn.ModuleList([Block(), Block()])
        self.norm = nn.LayerNorm(64)
        self.classifier = nn.Linear(64, 8)

    def forward(self, tokens, ablate=None):
        x = self.token(tokens) + self.position(torch.arange(tokens.shape[1], device=tokens.device))
        maps = []
        for layer, block in enumerate(self.blocks):
            head = ablate[1] if ablate is not None and ablate[0] == layer else None
            x, attention = block(x, head)
            maps.append(attention)
        logits = self.classifier(self.norm(x))
        return logits, maps
