"""Tests for CascadedESModel (model_cascade.py)."""

import pytest
import torch

from training.model_cascade import (
    MusicalTimeEmbedding,
    CascadedESModel,
    make_causal_mask,
)


class TestMusicalTimeEmbedding:
    def test_output_shape(self):
        emb = MusicalTimeEmbedding(d_model=64)
        times = torch.tensor([[0.0, 1.0, 2.0, 3.0]])  # (1, 4)
        out = emb(times)
        assert out.shape == (1, 4, 64)

    def test_different_times_produce_different_embeddings(self):
        emb = MusicalTimeEmbedding(d_model=64)
        times = torch.tensor([[0.0, 10.0]])
        out = emb(times)
        # Embeddings at t=0 and t=10 should differ
        assert not torch.allclose(out[0, 0], out[0, 1])

    def test_batch_independence(self):
        emb = MusicalTimeEmbedding(d_model=32)
        times = torch.tensor([[0.0, 1.0], [5.0, 6.0]])
        out = emb(times)
        assert out.shape == (2, 2, 32)
        # Same time values should give same embedding
        t1 = torch.tensor([[0.0]])
        t2 = torch.tensor([[0.0]])
        assert torch.allclose(emb(t1), emb(t2))

    def test_zero_time(self):
        emb = MusicalTimeEmbedding(d_model=16)
        times = torch.zeros(1, 1)
        out = emb(times)
        # sin(0) = 0 for all dims, cos(0) = 1 for all dims
        half = out.shape[-1] // 2
        assert torch.allclose(out[0, 0, :half], torch.zeros(half), atol=1e-6)
        assert torch.allclose(out[0, 0, half:], torch.ones(half), atol=1e-6)


class TestCausalMask:
    def test_shape(self):
        mask = make_causal_mask(5, torch.device("cpu"))
        assert mask.shape == (5, 5)

    def test_upper_triangular(self):
        mask = make_causal_mask(4, torch.device("cpu"))
        # diagonal should be False (can attend to self)
        for i in range(4):
            assert not mask[i, i].item()
        # upper triangle should be True (blocked)
        assert mask[0, 1].item()
        assert mask[0, 3].item()
        # lower triangle should be False (allowed)
        assert not mask[1, 0].item()
        assert not mask[3, 0].item()


class TestCascadedESModel:
    @pytest.fixture
    def small_model(self):
        return CascadedESModel(
            pad_id=0,
            type_names=["TIME_SHIFT", "BAR", "INST", "VEL", "PITCH_GENERAL", "DUR",
                         "SEP", "CHORD_ROOT", "CHORD_QUAL"],
            head_sizes=[96, 16, 6, 8, 60, 17, 1, 12, 5],
            num_embeddings=250,
            d_model=64,
            n_heads=4,
            n_layers=2,
            ff_mult=2,
            dropout=0.0,
        )

    def test_output_shapes(self, small_model):
        B, T = 2, 32
        x = torch.randint(1, 200, (B, T))
        times = torch.rand(B, T) * 20.0
        type_logits, value_logits = small_model(x, times)

        assert type_logits.shape == (B, T, 9)
        assert len(value_logits) == 9
        assert value_logits[0].shape == (B, T, 96)
        assert value_logits[-1].shape == (B, T, 5)

    def test_different_musical_times_give_different_outputs(self, small_model):
        B, T = 1, 16
        x = torch.randint(1, 200, (B, T))
        times_a = torch.zeros(B, T)
        times_b = torch.ones(B, T) * 100.0

        type_a, _ = small_model(x, times_a)
        type_b, _ = small_model(x, times_b)
        # Different times should produce different logits
        assert not torch.allclose(type_a, type_b, atol=1e-4)

    def test_padding_handled(self, small_model):
        B, T = 2, 16
        x = torch.randint(1, 200, (B, T))
        x[1, 10:] = 0  # pad second sequence
        times = torch.rand(B, T) * 10.0
        type_logits, value_logits = small_model(x, times)
        # Should not crash
        assert type_logits.shape == (B, T, 9)

    def test_variable_batch_lengths(self, small_model):
        """Different sequence lengths within a batch (padded)."""
        x = torch.tensor([
            [1, 2, 3, 0, 0],
            [1, 2, 3, 4, 5],
        ])
        times = torch.tensor([
            [0.0, 0.5, 1.0, 0.0, 0.0],
            [0.0, 0.5, 1.0, 1.5, 2.0],
        ])
        type_logits, value_logits = small_model(x, times)
        assert type_logits.shape == (2, 5, 9)

    def test_gradients_flow(self, small_model):
        """Verify gradients flow through the model."""
        B, T = 2, 16
        x = torch.randint(1, 200, (B, T))
        times = torch.rand(B, T) * 10.0

        type_logits, value_logits = small_model(x, times)
        loss = type_logits.sum() + sum(v.sum() for v in value_logits)
        loss.backward()

        for name, param in small_model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
