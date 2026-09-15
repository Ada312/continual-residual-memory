# Checkpoints

The small self-trained CRM files are kept for immediate load tests. Source/backbone weights remain external. Verify downloaded checkpoint hashes before inference; neither filenames nor architecture names alone identify a final model.

| Purpose | Expected file / source | SHA256 | In repository |
| --- | --- | --- | --- |
| Frozen CMGAN Source | Official SETTA base `08ea624f` tracked `checkpoints/cmgan_ears.th`; pass via `--cmgan-checkpoint` | `2649bc63511f6c59bf2340f6ab2305c9c3f5fbe1aa949434c176ede3ec65108f` | No |
| Utterance-local recovery | `checkpoints/crm/static_best.th` | `69c19970b7000b09d1b611b0dbc8786c24c9d9ca1c788cc077ff551225030523` | Yes |
| Memory-conditioned refinement | `checkpoints/crm/dynamic_best.th` | `17d876b7a69d83b93dd335d3bad7c35bd4fda641f8176edb64094b974592be21` | Yes |
| LaDen EARS foundation map | `checkpoints/baselines/WavLM_EARS_map.th` | `3f2102adb72c406cd34ad212d76b19db56e53bec2658b1bfb861fda1a1708963` | Yes |
| WavLM Large (LaDen) | `microsoft/wavlm-large`, revision `c1423ed94bb01d80a3f5ce5bc39f6026a0f4828c` | cached weight blob `fdee460e529396ddb2f8c8e8ce0ad74cfb747b726bc6f612e666c7c1e1963c9d` | No |

Cross-backbone Source checkpoint identities, five CRM checkpoint pairs, commits and exact inference settings are listed in [CROSS_BACKBONE.md](CROSS_BACKBONE.md). `checkpoints/crm/config.json` retains the three original projector construction keys required by frozen Stage 2 training and also carries final memory settings for the efficiency loader; `configs/cmgan/final.json` is the public deployment config. The final refinement checkpoint includes inactive `memory_adapter` weights for backward-compatible strict loading. Do not count them as active refinement parameters.
