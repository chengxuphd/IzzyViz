"""Shared fixtures.

Matplotlib is forced onto the non-interactive Agg backend before any test
imports izzyviz, so the suite runs headless in CI without a display.
"""
import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
import torch


@pytest.fixture
def rng():
    """Seeded generator so every test is deterministic."""
    return np.random.default_rng(20260828)


@pytest.fixture
def attention(rng):
    """A small, row-normalised 8x8 attention matrix as a torch tensor."""
    raw = torch.tensor(rng.random((8, 8)), dtype=torch.float32)
    return torch.softmax(raw, dim=-1)


@pytest.fixture
def layered_attention(rng):
    """HuggingFace-style output: one (batch=1, heads=2, 8, 8) tensor per layer."""
    return tuple(
        torch.softmax(torch.tensor(rng.random((1, 2, 8, 8)), dtype=torch.float32), dim=-1)
        for _ in range(2)
    )


@pytest.fixture
def tokens():
    return [f"tok{i}" for i in range(8)]
