from __future__ import annotations

from typing import Sequence

import torch


class SyntheticDialectWorld:
    """Compositional latent dialect environment supporting OOD domains and Zero-Shot OOV classes.

    When ``compositional=True`` (default):
    - Uses a compact active syllable pool ``S`` (default ``6``, with ``S^num_segments >> class_count``)
      and a Latin-square prefix so the first ``S`` classes cover 100% of all active syllables at
      every segment position ``0 .. num_segments - 1``.
    - Ensures every token in every OOV validation class has been acoustically observed during
      training (zero unseen acoustic phonemes).
    - Sets ``max_tokens = num_segments`` so every text token maps 1-to-1 to a synthesized
      acoustic segment (zero silent trailing suffix tokens acting as class-hash shortcuts).
    """

    def __init__(
        self,
        input_dim: int = 64,
        vocabulary_size: int = 128,
        class_count: int = 8,
        domain_count: int = 3,
        num_segments: int = 3,
        max_tokens: int | None = None,
        seed: int = 17,
        compositional: bool = True,
        syllable_vocab_size: int | None = None,
    ):
        if vocabulary_size < 4:
            raise ValueError("vocabulary_size must be >= 4")
        if num_segments < 1:
            raise ValueError("num_segments must be >= 1")
        self.input_dim = input_dim
        self.vocabulary_size = vocabulary_size
        self.class_count = class_count
        self.domain_count = max(1, domain_count)
        self.num_segments = num_segments
        self.compositional = bool(compositional)
        # In compositional mode, match token length strictly to acoustic segments (zero silent suffixes).
        if self.compositional:
            self.max_tokens = num_segments
        else:
            self.max_tokens = max(num_segments, max_tokens or 6)
        self.seed = seed

        generator = torch.Generator(device="cpu").manual_seed(seed)

        # Choose a compact syllable pool S so S^num_segments >> class_count and train_classes cover 100% of S.
        if syllable_vocab_size is not None:
            active_syllables = int(syllable_vocab_size)
        else:
            # Smallest S >= 4 (default 6 when feasible) with S^num_segments >= max(4 * class_count, class_count + 1)
            active_syllables = 6
            while (active_syllables ** num_segments) < max(class_count * 2, class_count + 1):
                active_syllables += 1
        active_syllables = max(2, min(active_syllables, vocabulary_size - 1))
        if (active_syllables ** num_segments) < class_count:
            raise ValueError(
                f"syllable_vocab_size={active_syllables} with num_segments={num_segments} "
                f"cannot represent class_count={class_count} unique words"
            )
        self.active_syllables = active_syllables

        self.token_acoustic_bank = torch.randn(
            vocabulary_size, input_dim, generator=generator
        ) * 1.5
        self.token_acoustic_bank[0] = 0.0  # padding token has zero acoustic energy

        self.domain_offsets = torch.randn(
            self.domain_count, input_dim, generator=generator
        ) * 0.35

        # Construct strictly unique ordered sequences of length max_tokens over {1..S}.
        # First S classes form a Latin square over positions 0..num_segments-1 so every syllable
        # appears at every position within the first S training classes.
        sequences: list[tuple[int, ...]] = []
        seen: set[tuple[int, ...]] = set()
        for cls_idx in range(min(class_count, active_syllables)):
            seq = tuple(((cls_idx + pos) % active_syllables) + 1 for pos in range(self.max_tokens))
            sequences.append(seq)
            seen.add(seq)

        if len(sequences) < class_count:
            # Deterministically sample remaining unique ordered recombinations of the same S syllables.
            # Enumerate via mixed-radix stride coprime to S^max_tokens for even syllable distribution.
            total_space = active_syllables ** self.max_tokens
            stride = 1
            for candidate_stride in (7, 11, 13, 17, 19, 23, 29, 31):
                if total_space % candidate_stride != 0:
                    stride = candidate_stride
                    break
            idx = 0
            while len(sequences) < class_count:
                code = (idx * stride + 1) % total_space
                idx += 1
                digits: list[int] = []
                rem = code
                for _ in range(self.max_tokens):
                    digits.append((rem % active_syllables) + 1)
                    rem //= active_syllables
                seq = tuple(digits)
                if seq not in seen:
                    seen.add(seq)
                    sequences.append(seq)

        self.class_token_sequences = torch.tensor(sequences, dtype=torch.long)

        if self.compositional:
            # Every acoustic segment 0..num_segments-1 corresponds 1-to-1 to class_token_sequences[:, seg].
            coarticulation = torch.randn(
                class_count, num_segments, input_dim, generator=generator
            ) * 0.15
            seg_tokens = self.class_token_sequences[:, :num_segments]
            self.class_prototypes = self.token_acoustic_bank[seg_tokens] + coarticulation
        else:
            self.class_prototypes = torch.randn(
                class_count, num_segments, input_dim, generator=generator
            ) * 1.5

    def verify_oov_syllable_coverage(
        self,
        train_classes: Sequence[int],
        val_classes: Sequence[int],
        per_position: bool = False,
    ) -> bool:
        """Return True iff every syllable used in ``val_classes`` (pos < num_segments) appears in ``train_classes``."""
        train_seqs = self.class_token_sequences[list(train_classes), : self.num_segments]
        val_seqs = self.class_token_sequences[list(val_classes), : self.num_segments]
        if per_position:
            for pos in range(self.num_segments):
                train_set = set(train_seqs[:, pos].tolist())
                val_set = set(val_seqs[:, pos].tolist())
                if not val_set.issubset(train_set):
                    return False
            return True
        train_all = set(train_seqs.reshape(-1).tolist())
        val_all = set(val_seqs.reshape(-1).tolist())
        return val_all.issubset(train_all)

    def canonical_gallery(
        self,
        device: str = "cpu",
        allowed_classes: Sequence[int] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return deduplicated canonical text entries with zero silent trailing tokens."""
        target_device = torch.device(device)
        if allowed_classes is not None:
            cls_ids = torch.tensor(list(allowed_classes), dtype=torch.long)
        else:
            cls_ids = torch.arange(self.class_count, dtype=torch.long)
        tokens = self.class_token_sequences[cls_ids].clone()
        mask = torch.ones_like(tokens, dtype=torch.bool)
        return {
            "text_tokens": tokens.to(target_device),
            "text_mask": mask.to(target_device),
            "classes": cls_ids.to(target_device),
        }

    def sample_batch(
        self,
        batch_size: int = 16,
        max_frames: int = 24,
        max_tokens: int | None = None,
        seed: int = 17,
        device: str = "cpu",
        noise_std: float = 0.35,
        allowed_domains: Sequence[int] | None = None,
        allowed_classes: Sequence[int] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Sample a deterministic batch with 1-to-1 acoustic-segment-to-token alignment."""
        if self.compositional:
            # Strict 1-to-1 alignment between num_segments acoustic states and text tokens.
            resolved_max_tokens = self.num_segments
            min_tokens = self.num_segments
        else:
            resolved_max_tokens = max(max_tokens or self.max_tokens, self.num_segments)
            min_tokens = min(self.num_segments, resolved_max_tokens)

        generator = torch.Generator(device="cpu").manual_seed(seed)
        min_frames = min(max_frames, max(6, self.num_segments * 2))
        frame_lengths = torch.randint(min_frames, max_frames + 1, (batch_size,), generator=generator)
        token_lengths = torch.randint(
            min_tokens, min(resolved_max_tokens, self.max_tokens) + 1, (batch_size,), generator=generator
        )
        frame_mask = torch.arange(max_frames).unsqueeze(0) < frame_lengths.unsqueeze(1)
        text_mask = torch.arange(resolved_max_tokens).unsqueeze(0) < token_lengths.unsqueeze(1)

        if allowed_classes is not None:
            class_pool = torch.tensor(list(allowed_classes), dtype=torch.long)
            if class_pool.numel() == 0:
                raise ValueError("allowed_classes must be non-empty")
            if int(class_pool.min().item()) < 0 or int(class_pool.max().item()) >= self.class_count:
                raise ValueError(f"allowed_classes out of range [0, {self.class_count - 1}]")
            cls_choice = torch.randint(0, class_pool.numel(), (batch_size,), generator=generator)
            classes = class_pool[cls_choice]
        else:
            classes = torch.randint(0, self.class_count, (batch_size,), generator=generator)

        if allowed_domains is not None:
            domain_pool = torch.tensor(list(allowed_domains), dtype=torch.long)
            if domain_pool.numel() == 0:
                raise ValueError("allowed_domains must be non-empty")
            if int(domain_pool.min().item()) < 0 or int(domain_pool.max().item()) >= self.domain_count:
                raise ValueError(f"allowed_domains out of range [0, {self.domain_count - 1}]")
            dom_choice = torch.randint(0, domain_pool.numel(), (batch_size,), generator=generator)
            domains = domain_pool[dom_choice]
        else:
            domains = torch.randint(0, self.domain_count, (batch_size,), generator=generator)

        frames = torch.randn(batch_size, max_frames, self.input_dim, generator=generator) * noise_std
        for index in range(batch_size):
            valid_len = int(frame_lengths[index].item())
            cls_id = int(classes[index].item())
            dom_id = int(domains[index].item())
            for seg in range(self.num_segments):
                start = (seg * valid_len) // self.num_segments
                end = ((seg + 1) * valid_len) // self.num_segments
                if end > start:
                    t_ramp = torch.linspace(0.85, 1.15, end - start).unsqueeze(-1)
                    frames[index, start:end] += (
                        self.class_prototypes[cls_id, seg].unsqueeze(0) * t_ramp
                        + self.domain_offsets[dom_id].unsqueeze(0)
                    )
            if valid_len < max_frames:
                frames[index, valid_len:] = 0.0

        tokens = torch.zeros((batch_size, resolved_max_tokens), dtype=torch.long)
        for index in range(batch_size):
            valid_tok = int(token_lengths[index].item())
            cls_id = int(classes[index].item())
            tokens[index, :valid_tok] = self.class_token_sequences[cls_id, :valid_tok]

        target_device = torch.device(device)
        return {
            "frames": frames.to(target_device),
            "frame_mask": frame_mask.to(target_device),
            "text_tokens": tokens.to(target_device),
            "text_mask": text_mask.to(target_device),
            "classes": classes.to(target_device),
            "domains": domains.to(target_device),
        }


def make_synthetic_batch(
    batch_size: int = 16,
    input_dim: int = 64,
    model_dim: int | None = None,
    vocabulary_size: int = 128,
    class_count: int = 8,
    domain_count: int = 3,
    max_frames: int = 24,
    max_tokens: int | None = None,
    seed: int = 17,
    device: str = "cpu",
    world: SyntheticDialectWorld | None = None,
    allowed_domains: Sequence[int] | None = None,
    allowed_classes: Sequence[int] | None = None,
    compositional: bool = True,
    syllable_vocab_size: int | None = None,
) -> dict[str, torch.Tensor]:
    """Create deterministic dense-alignment fixtures from a shared or fresh compositional world."""
    del model_dim
    active_world = world or SyntheticDialectWorld(
        input_dim=input_dim,
        vocabulary_size=vocabulary_size,
        class_count=class_count,
        domain_count=domain_count,
        max_tokens=max_tokens,
        seed=seed,
        compositional=compositional,
        syllable_vocab_size=syllable_vocab_size,
    )
    return active_world.sample_batch(
        batch_size=batch_size,
        max_frames=max_frames,
        max_tokens=max_tokens,
        seed=seed,
        device=device,
        allowed_domains=allowed_domains,
        allowed_classes=allowed_classes,
    )
