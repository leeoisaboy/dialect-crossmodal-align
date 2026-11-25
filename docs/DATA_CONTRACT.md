# Input and Data Contract (v5)

The public repository contains no licensed raw audio, no private speaker tables,
and no offline forced-alignment scripts. The converged 4-stage engine operates
strictly on standard **50 Hz** (`20 ms/frame`) frozen speech representations
paired with language-agnostic phonetic/IPA/subword token sequences.

## 1. Real 50 Hz Tier Directory Layout (`RealAlignmentDataset`)

Each benchmark tier directory (`T1_wu`, `T2_kespeech`, `T3_libriphrase`, or any
custom language/dialect corpus) provides 4 standard files:

- `features_50hz.fp16.npy`: flat `[Total_Frames, feature_dim]` frozen 50 Hz SSL frame array (e.g. `WavLM-Large`, `feature_dim = 1024`).
- `feature_offsets.int64.npy`: `[N_clips + 1]` cumulative frame boundaries per clip.
- `manifest.jsonl`: one JSON record per clip (`idx`, `clip_id`, `split`, `word`, `class_id`, `is_oov`, `is_negative`, `units`, `token_ids`, `speaker`, `domain`, `domain_name`, `speed_factor`, `duration_s`, `num_frames`).
- `metadata.json`: vocabulary map (`unit_vocab`, `0` reserved for `<pad>`), `word_gallery`, `feature_dim`, `n_train_words`, `n_oov_words`.

## 2. Batch Tensor Contract

| Key | Shape | Meaning |
| --- | --- | --- |
| `frames` | `[batch, time, input_dim]` | frozen 50 Hz acoustic states (`20 ms/frame`) |
| `frame_mask` | `[batch, time]` | boolean valid-frame mask |
| `text_tokens` | `[batch, query_time]` | ordered phonetic / IPA / subword token IDs (`0` = `<pad>`) |
| `text_mask` | `[batch, query_time]` | boolean valid-token mask |
| `classes` | `[batch]` | word/class ID (`-1` for open-set non-keyword negatives) |
| `domains` | `[batch]` | dialect / channel domain label |

## 3. Verified Zero-Leakage Invariants (`verify_dataset_invariants`)

Every real-data tier must satisfy all four automated assertions before training:
1. `train_words ∩ oov_words = ∅` (zero word overlap)
2. `train_speakers ∩ test_speakers = ∅` (zero speaker overlap)
3. `100.0%` of OOV phonetic/subword units appear in `train_words`
4. Differential frame stride matches `50.0 Hz ± 1.0 Hz`
