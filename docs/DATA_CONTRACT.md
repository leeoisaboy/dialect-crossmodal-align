# Input and data contract (v4)

The public repository contains no licensed audio, no speaker tables, and no
derived private encoder states. Training starts from a compositional
`SyntheticDialectWorld` (supporting OOD `allowed_domains` and Zero-Shot OOV
`allowed_classes` splits via a shared syllable `token_acoustic_bank`), or from
a private tensor bundle with these keys:

| Key | Shape | Meaning |
| --- | --- | --- |
| `frames` | `[batch, time, input_dim]` | frozen acoustic states |
| `frame_mask` | `[batch, time]` | boolean valid-frame mask (may be non-contiguous or all-False) |
| `text_tokens` | `[batch, query_time]` | ordered syllable/character token IDs (`0` reserved for padding) |
| `text_mask` | `[batch, query_time]` | boolean valid-token mask |
| `classes` | `[batch]` | word/class ID for CosFace, class-conditional domain alignment, and multi-positive retrieval |
| `domains` | `[batch]` | enrollment/search, speaker, or environment label for DANN and domain alignment |
