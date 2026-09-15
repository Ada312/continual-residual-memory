# Continual Residual Memory

Official implementation of **Continual Residual Memory for Gradient-Free Test-Time Adaptation in Speech Enhancement**.

CRM augments a frozen speech enhancement backbone with utterance-local residual recovery and causal prototype-memory refinement. At test time, network parameters stay frozen: a query reads only the previous prototype state, the current utterance is enhanced, and its residual observation is written **after** the output. This is utterance-level adaptation, not frame-streaming or test-time optimization.

## Method

```text
Noisy waveform y_t -> frozen backbone estimate x_hat_t -> residual r_t
  -> seven-channel Phi_t -> Utterance-Local Residual Recovery (P_s, H_t, g_static)
  -> no-memory estimate X_hat_t^static
  -> read Prototype Memory M_(t-1) with query u_t -> Memory Readout Z_t
  -> Basis Branch B_t(H_t) x Coordinate Branch C_t(Z_t)
  -> signed low-rank delta_g -> bounded g_dyn -> enhanced waveform
  -> occupancy q_(t,tau) -> write observation o_t -> M_t
```

Implementation: [`crm/residual.py`](crm/residual.py), [`crm/recovery.py`](crm/recovery.py), [`crm/memory.py`](crm/memory.py), [`crm/refinement.py`](crm/refinement.py), and [`crm/model.py`](crm/model.py). The final class bodies were extracted from the frozen release candidate; model weights, state-dict attribute names, memory matching and update calculations are unchanged. [`docs/PAPER_CODE_NOTATION.md`](docs/PAPER_CODE_NOTATION.md) maps every paper symbol to the exact implementation.

## Installation

Python 3.9 and the pinned [environment](environment.yml) are the recorded reproduction target. An existing compatible PyTorch environment can use:

```bash
python -m pip install -r requirements-core.txt
```

Install [baseline](requirements-baselines.txt) and [cross-backbone](requirements-cross-backbone.txt) dependencies only for those experiments. Third-party backbones are optional when importing `crm`.

For repository checks, install `requirements-dev.txt`, then run
`ruff check .` and `python -m pytest -q tests`.

## External Resources

Third-party resources are not redistributed in this repository unless a
specific vendored compatibility component is identified in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). Download datasets,
upstream source trees, and backbone weights from their official providers and
follow their respective licenses and terms of use. Repository paths and pinned
versions are documented in [`THIRD_PARTY.md`](THIRD_PARTY.md).

## Datasets

| Resource | Official source | Use in this paper | Citation |
| --- | --- | --- | --- |
| EARS / EARS-WHAM | [EARS](https://github.com/facebookresearch/ears_dataset), [EARS-WHAM generator](https://github.com/sp-uhh/ears_benchmark) | Only offline CRM training source | [Richter et al., 2024](https://arxiv.org/abs/2406.06185) |
| WHAM! | [Official site](http://wham.whisper.ai/) | Noise for EARS-WHAM | [Wichern et al., 2019](https://doi.org/10.1109/ICASSP.2019.8683555) |
| DNS Challenge | [Official repository](https://github.com/microsoft/DNS-Challenge) | DNS 2020 synthetic no-reverb evaluation | [Reddy et al., 2020](https://doi.org/10.21437/Interspeech.2020-1768) |
| DEMAND | [Official Zenodo record](https://doi.org/10.5281/zenodo.1227121) | Noise used to construct EARS-D | [Thiemann et al., 2013](https://doi.org/10.1121/1.4799597) |
| LibriSpeech | [OpenSLR 12](https://www.openslr.org/12) | Clean speech for Libri-MUSAN evaluation mixtures | [Panayotov et al., 2015](https://doi.org/10.1109/ICASSP.2015.7178964) |
| MUSAN | [OpenSLR 17](https://www.openslr.org/17) | Music noise and old-domain memory stream | [Snyder et al., 2015](https://arxiv.org/abs/1510.08484) |

Training uses **only EARS-WHAM** (8,192 training and 632 speaker-disjoint held-out mixtures). The paper evaluates three separate empty-memory-start streams: DNS 2020 synthetic no-reverb (150), EARS-D (886), and Libri-MUSAN music (2,620). Dataset acquisition and preparation scripts are in [`data/`](data/) and [`docs/DATA.md`](docs/DATA.md). `manifests/` contains portable frozen identities and order; it contains no audio. Clean references are used for source-domain offline training and final evaluation, never for target-domain memory updates.

## Pretrained Backbones

| Resource | Official source | Use in this paper | Citation |
| --- | --- | --- | --- |
| CMGAN | [Official repository](https://github.com/ruizhecao96/CMGAN) | Main frozen enhancement backbone | [Cao et al., 2022](https://doi.org/10.21437/Interspeech.2022-517) |
| StoRM | [Official repository](https://github.com/sp-uhh/storm) | StoRM-50 cross-backbone evaluation | [Lemercier et al., 2023](https://doi.org/10.1109/TASLP.2023.3294692) |
| FlowSE | [Official repository](https://github.com/seongq/flowmse) | Flow-matching cross-backbone evaluation | [Lee et al., 2025](https://doi.org/10.1109/ICASSP49660.2025.10888274) |
| GTCRN | [Official repository](https://github.com/Xiaobin-Rong/gtcrn) | Discriminative cross-backbone evaluation | [Rong et al., 2024](https://ieeexplore.ieee.org/document/10448310) |
| FastEnhancer-B | [Official repository](https://github.com/aask1357/fastenhancer) | Streaming cross-backbone evaluation | [Ahn et al., 2026](https://arxiv.org/abs/2509.21867) |
| UL-UNAS | [Official repository](https://github.com/Xiaobin-Rong/ul-unas) | Ultra-lightweight cross-backbone evaluation | [Rong et al., 2026](https://doi.org/10.1109/TASLPRO.2026.3661271) |
| SETTA / LaDen / MPol | [Official repository](https://github.com/tobiaaa/SETTA) | Gradient-based TTA baselines | [LaDen](https://arxiv.org/abs/2509.04280), [MPol](https://arxiv.org/abs/2601.14770) |
| WavLM Large | [Official model](https://huggingface.co/microsoft/wavlm-large) | Frozen LaDen embedding encoder | [Chen et al., 2022](https://arxiv.org/abs/2110.13900) |

CMGAN is the main frozen backbone. Five independently trained CRM checkpoint
pairs are provided for the recorded cross-backbone artifacts. Small
self-trained CRM checkpoints are included; obtain CMGAN and the other upstream
checkpoint files separately and check their hashes in
[`docs/CHECKPOINTS.md`](docs/CHECKPOINTS.md). Upstream repository commits and
exact inference options are in
[`docs/CROSS_BACKBONE.md`](docs/CROSS_BACKBONE.md).

LaDen's third-party EARS foundation map is also external. Download and verify
it from SETTA commit `08ea624f` into
`checkpoints/baselines/WavLM_EARS_map.th` using
[`checkpoints/baselines/README.md`](checkpoints/baselines/README.md); the
expected SHA256 is
`3f2102adb72c406cd34ad212d76b19db56e53bec2658b1bfb861fda1a1708963`.
The FlowSE experiment uses Lee et al.'s `seongq/flowmse` at commit
`f6b479d13fecc6cb6f12394f46dfc6799fb479b6` and its official
VoiceBank-DEMAND checkpoint. The upstream code and checkpoint remain external;
their exact acquisition, hash and inference settings are recorded in
[`docs/CROSS_BACKBONE.md`](docs/CROSS_BACKBONE.md).

## Training

Both stages use 16 kHz audio, 512-point Hann STFT/128-sample hop, 2-s crops, batch size 8, AdamW with weight decay `1e-4`, and cosine scheduling. The Stage 1 projector is trained for 4 epochs at `3e-4` and frozen. Stage 2 optimizes only the Basis/Coordinate Branches for 3 epochs at `1e-3` with grouped EARS-WHAM histories, `K_train=2`, memory dropout 0.20, and the paper counterfactual objective. Full parameters and runnable commands are in [`docs/REPRODUCTION.md`](docs/REPRODUCTION.md).

```bash
python scripts/train_recovery.py --help
python scripts/train_refinement.py --help
```

The public training entries invoke the paper-aligned implementations under `crm/_training/`. Historical checkpoint attributes (`input`, `blocks`, `output`, `delta_basis`, `delta_coordinates`, and the inactive `memory_adapter`) remain load-compatible. Deployment uses `K_max=64` and readout mass 0.95; the frozen **training** config has `readout_mass=null` for `K_train=2`. This distinction is recorded without modifying either procedure.

## Evaluation

| Paper experiment | Entry | Data/outputs |
| --- | --- | --- |
| Main comparison: Backbone, LaDen, MPol, CRM (no memory), CRM | `scripts/eval_main.py`, `scripts/eval_baselines.py` | DNS / EARS-D / Libri-MUSAN; seven shared metrics |
| Continual progression (PESQ/COVL) | `scripts/plot_continual.py` | Frozen DNS 150 per-file CSVs; trailing causal 10-utterance mean |
| Libri-MUSAN to DNS transition | `scripts/eval_cross_domain.py`, `scripts/aggregate_cross_domain.py` | 2,620 post-inference writes, keep state, then 150 DNS writes |
| Cross-backbone applicability | `scripts/eval_cross_backbone.py` | Five upstream backbones, DNS 150, backbone-specific CRM checkpoints |
| Parameters, RTF, registered-op FLOPs, storage | `scripts/count_parameters.py`, `scripts/measure_rtf.py`, `scripts/measure_flops.py`, `scripts/estimate_memory_storage.py` | Frozen protocol; see [`docs/EFFICIENCY.md`](docs/EFFICIENCY.md) |

Run `python scripts/plot_continual.py` to regenerate the manuscript's two plots from the included CSVs, without waveform inference or metric recomputation. Reference images are under `results/continual/`. All commands and input layout are documented in [`docs/REPRODUCTION.md`](docs/REPRODUCTION.md).

## Paper-to-Code Correspondence

The [notation map](docs/PAPER_CODE_NOTATION.md) follows the paper's three method subsections. Public entry names use *utterance-local recovery* and *memory-conditioned refinement*. `g_static`, `g_dyn`, `alpha_dyn`, and `delta_g` retain the paper's mathematical meaning. Filenames ending in `static` or `dynamic` in frozen result artifacts are preserved for source identity, not alternative model families.

## Citation

```bibtex
@misc{shan2026crm,
  title  = {Continual Residual Memory for Gradient-Free Test-Time Adaptation in Speech Enhancement},
  author = {Shan, Yijia and Wang, Tianrui and Wang, Zixiang and Wang, Yu and Chen, Xie},
  year   = {2026},
  url    = {https://github.com/Ada312/continual-residual-memory}
}
```

Venue and DOI fields should be added after publication metadata is finalized.

## License and Provenance

See [LICENSE](LICENSE), [`THIRD_PARTY.md`](THIRD_PARTY.md),
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md),
[`docs/IMPLEMENTATION_PROVENANCE.md`](docs/IMPLEMENTATION_PROVENANCE.md), and
the [release verification status](docs/RELEASE_STATUS.md). `LICENSE` applies
to project code and the identified GPL-compatible derivatives; retained
EARS-WHAM selection metadata remains under the upstream CC BY-NC 4.0 terms
documented in [`manifests/training/README.md`](manifests/training/README.md).
The repository contains no original `.git` history, benchmark audio, full
waveform outputs, large upstream weights, training caches, or private audit
dumps.
