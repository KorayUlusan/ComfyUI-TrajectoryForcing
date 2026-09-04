"""Translating an allocator failure into something a user can act on.

An XLA out-of-memory error is a page of buffer arithmetic and assignment
tables. It is precise, and it never says what to do. On a card with room for
this model it almost always means another node is holding VRAM, which is a
thing the reader can fix -- so that is what the message should say.

Matching is on the message rather than the exception class, deliberately. The
alternative is importing jax here to catch `XlaRuntimeError`, which initialises
the GPU backend at import time and is exactly what the rest of this module goes
out of its way to avoid.
"""
from __future__ import annotations

import pytest

from tf_nodes.pipeline import _OOM_SIGNS, _vram_advice


class XlaRuntimeError(RuntimeError):
    """Stand-in with the shape jaxlib's has: the word is in the message."""


class TestAnOomBecomesAdvice:
    @pytest.mark.parametrize(
        "exc",
        [
            XlaRuntimeError("RESOURCE_EXHAUSTED: Out of memory allocating 2147483648 bytes."),
            RuntimeError("CUDA out of memory. Tried to allocate 512.00 MiB"),
            RuntimeError("CUDA_ERROR_OUT_OF_MEMORY: out of memory"),
            RuntimeError("Failed to allocate request for 8.00GiB on device ordinal 0"),
            MemoryError("XlaRuntimeError RESOURCE_EXHAUSTED"),
        ],
    )
    def test_every_shape_of_full_card_is_recognised(self, exc):
        with pytest.raises(RuntimeError) as info:
            with _vram_advice("Generating"):
                raise exc
        assert "ran out of GPU memory" in str(info.value)

    def test_the_advice_names_the_measured_numbers_and_the_levers(self):
        with pytest.raises(RuntimeError) as info:
            with _vram_advice("Decoding"):
                raise RuntimeError("RESOURCE_EXHAUSTED")
        text = str(info.value)
        assert "6.6 GiB" in text
        assert "TF_XLA_MEM_FRACTION" in text
        assert "only model" in text

    def test_it_says_which_operation_died(self):
        """Generate, resume and decode have different peaks; which one it was
        is the first thing worth knowing."""
        with pytest.raises(RuntimeError) as info:
            with _vram_advice("Resuming from an edited level"):
                raise RuntimeError("out of memory")
        assert "Resuming from an edited level" in str(info.value)

    def test_the_original_is_chained_not_swallowed(self):
        """A bug report still needs the allocator's own numbers."""
        original = RuntimeError("RESOURCE_EXHAUSTED: buffer table follows")
        with pytest.raises(RuntimeError) as info:
            with _vram_advice("Generating"):
                raise original
        assert info.value.__cause__ is original


class TestEverythingElseIsLeftAlone:
    """Rewriting an unrelated failure as a VRAM problem would send the reader
    somewhere there is nothing to find."""

    @pytest.mark.parametrize(
        "exc",
        [
            ValueError("TF_LEVELS expects [L,H,W,C], got (3, 16, 16)"),
            FileNotFoundError("No such TrajectoryForcing config: nope.yml"),
            KeyError("class_id"),
            ZeroDivisionError("division by zero"),
        ],
    )
    def test_an_unrelated_error_passes_through_unchanged(self, exc):
        with pytest.raises(type(exc)) as info:
            with _vram_advice("Generating"):
                raise exc
        assert info.value is exc

    def test_success_is_transparent(self):
        with _vram_advice("Generating"):
            result = 1 + 1
        assert result == 2

    def test_the_signs_do_not_match_ordinary_prose(self):
        """`_OOM_SIGNS` is matched case-insensitively against the whole message,
        so an over-broad entry would capture unrelated errors."""
        innocent = "Memory layout of the token grid is [H,W]; nothing is allocated here."
        assert not any(s.lower() in innocent.lower() for s in _OOM_SIGNS)
