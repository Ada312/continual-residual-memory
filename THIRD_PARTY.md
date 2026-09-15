# Third-Party Code and Data

| Component | Upstream and pinned identity | Role |
| --- | --- | --- |
| CMGAN | `https://github.com/ruizhecao96/CMGAN.git`, reference commit `39946005d9b66fa1e824edf4bb6bc9a06088e443` | Frozen Source architecture; release-candidate-compatible files under `backbones/cmgan/` |
| SETTA (LaDen/MPol) | `https://github.com/tobiaaa/SETTA.git`, base `08ea624f37dccc798f6bbffaf1f8f8e292c16e4b` | Exact paper baseline with `patches/setta_final_baselines.patch` |
| WavLM | `microsoft/wavlm-large`, revision recorded in `docs/CHECKPOINTS.md` | Frozen LaDen encoder |
| StoRM | `https://github.com/sp-uhh/storm.git`, `257e9636a7251ca40aa200753d5c0fe918e31879` | Frozen diffusion Source |
| GTCRN | `https://github.com/Xiaobin-Rong/gtcrn.git`, `502ebfab64da7c4a9af78dcb9c6ceef1ebb01c73` | Frozen Source |
| FastEnhancer | `https://github.com/aask1357/fastenhancer.git`, `f85223bd546b27f39dc0744e0310dcd246f750a4` | Frozen Source |
| FlowSE runtime | `https://github.com/seongq/flowmse.git`, `f6b479d13fecc6cb6f12394f46dfc6799fb479b6` | Frozen Source |
| UL-UNAS | `https://github.com/Xiaobin-Rong/ul-unas.git`, `00f7c700da43d38347f30a6ccebd86fcbc798e07` | Frozen Source |

Each upstream project, dataset, model weight, and optional package retains its own license and attribution terms. This repository carries no `.git` history, raw benchmark audio, or external backbone weights. `LICENSE` preserves the GPL-3.0 license of the original SETTA-derived research code. The user has stated that third-party/backbone/checkpoint publication permissions have been resolved; maintainers should retain that clearance record when publishing.
