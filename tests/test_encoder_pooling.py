"""Edge-case tests for masked temporal pooling."""

import torch

from failuredna.models.encoder import MaskedTemporalPooling


def test_last_pooling_uses_last_valid_position_for_left_padding():
    values = torch.tensor([[[10.0], [20.0], [30.0], [40.0]]])
    mask = torch.tensor([[0.0, 0.0, 1.0, 1.0]])

    pooled = MaskedTemporalPooling(method="last")(values, mask)

    assert torch.equal(pooled, torch.tensor([[40.0]]))


def test_last_pooling_returns_zero_for_fully_masked_sequence():
    values = torch.tensor([[[10.0], [20.0], [30.0]]])
    mask = torch.zeros((1, 3))

    pooled = MaskedTemporalPooling(method="last")(values, mask)

    assert torch.equal(pooled, torch.zeros((1, 1)))


def test_attention_pooling_returns_zero_for_fully_masked_sequence():
    values = torch.tensor([[[10.0, 1.0], [20.0, 2.0], [30.0, 3.0]]])
    mask = torch.zeros((1, 3))
    pooling = MaskedTemporalPooling(method="attention", d_model=2)

    pooled = pooling(values, mask)

    assert torch.isfinite(pooled).all()
    assert torch.equal(pooled, torch.zeros((1, 2)))
