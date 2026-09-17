# Continual Residual Memory for Gradient-Free Test-Time Adaptation in Speech Enhancement

[![Python 3.9](https://img.shields.io/badge/Python-3.9-3776AB.svg)](requirements.txt)
[![License: GPL-3.0](https://img.shields.io/badge/License-GPL--3.0-blue.svg)](LICENSE)

## Overview

Continual Residual Memory (CRM) is a gradient-free continual test-time adaptation framework for speech enhancement. It selectively recovers useful information from the residual discarded by a frozen enhancement backbone and refines the recovery using causal prototype memory from preceding utterances. During deployment, model parameters remain fixed while the memory state is updated online.

<p align="center">
  <img src="assets/crm_overview.png" alt="Overview of Continual Residual Memory" width="100%">
</p>
<p align="center"><em>Overview of Continual Residual Memory (CRM).</em></p>

## Installation

```bash
git clone https://github.com/Ada312/continual-residual-memory.git
cd continual-residual-memory

conda create -n crm python=3.9 -y
conda activate crm
pip install -r requirements.txt
```

## Checkpoints

The pretrained CMGAN checkpoint released by [SETTA](https://github.com/tobiaaa/SETTA) and the released CRM checkpoints are included:

| Component | Checkpoint |
| --- | --- |
| Frozen CMGAN backbone | `checkpoints/external/cmgan_ears.th` |
| Utterance-Local Residual Recovery | `checkpoints/crm/static_best.th` |
| Memory-Conditioned Refinement | `checkpoints/crm/dynamic_best.th` |

## Data Preparation

CRM is trained on EARS-WHAM and evaluated on DNS 2020 synthetic no-reverb, EARS-D, and Libri-MUSAN. Raw datasets are not redistributed.

Before training, prepare EARS-WHAM and generate the frozen CMGAN outputs:

```bash
export DATA_ROOT=/path/to/crm-data

python data/prepare_ears_wham_from_benchmark.py \
  --ears-wham-root /path/to/EARS-WHAM \
  --out-dir "$DATA_ROOT/ears_wham"

python backbones/cmgan/inference.py \
  --noisy-dir "$DATA_ROOT/ears_wham/noisy" \
  --output-dir "$DATA_ROOT/ears_wham/source" \
  --checkpoint checkpoints/external/cmgan_ears.th
```

See [`data/README.md`](data/README.md) for complete preparation instructions for EARS-WHAM, DNS, EARS-D, and Libri-MUSAN.

## Training

### Stage 1: Utterance-Local Residual Recovery

After completing the data preparation above, Stage 1 trains an utterance-local residual recovery module on top of the frozen CMGAN backbone.

```bash
python scripts/train_recovery.py \
  --config configs/cmgan/recovery_training.json \
  --clean-dir "$DATA_ROOT/ears_wham/clean" \
  --noisy-dir "$DATA_ROOT/ears_wham/noisy" \
  --source-dir "$DATA_ROOT/ears_wham/source" \
  --output-dir outputs/train_recovery
```

### Stage 2: Memory-Conditioned Refinement

Stage 2 freezes the Stage 1 recovery module and trains the memory-conditioned refinement module using causal source-domain histories.

```bash
python scripts/train_refinement.py \
  --config configs/cmgan/refinement_training.json \
  --clean-dir "$DATA_ROOT/ears_wham/clean" \
  --noisy-dir "$DATA_ROOT/ears_wham/noisy" \
  --source-dir "$DATA_ROOT/ears_wham/source" \
  --metadata "$DATA_ROOT/ears_wham/metadata.txt" \
  --recovery-checkpoint outputs/train_recovery/best.th \
  --context-cache outputs/cache/ears_wham_k2_grouped.pt \
  --output-dir outputs/train_refinement
```

## Inference and Evaluation

Use the checkpoints produced by Stage 1 and Stage 2 to enhance a target-domain stream:

```bash
python scripts/eval_main.py \
  --dataset dns \
  --noisy-dir "$DATA_ROOT/dns/noisy" \
  --clean-dir "$DATA_ROOT/dns/clean" \
  --cmgan-checkpoint checkpoints/external/cmgan_ears.th \
  --recovery-checkpoint outputs/train_recovery/best.th \
  --refinement-checkpoint outputs/train_refinement/best.th \
  --output-dir outputs/dns
```

Alternatively, use the released CRM checkpoints in `checkpoints/crm/` for direct evaluation without retraining:

```bash
python scripts/eval_main.py \
  --dataset dns \
  --noisy-dir "$DATA_ROOT/dns/noisy" \
  --clean-dir "$DATA_ROOT/dns/clean" \
  --cmgan-checkpoint checkpoints/external/cmgan_ears.th \
  --recovery-checkpoint checkpoints/crm/static_best.th \
  --refinement-checkpoint checkpoints/crm/dynamic_best.th \
  --output-dir outputs/dns
```

The same entry supports `dns`, `ears_d`, and `libri_musan`. Each dataset is processed as an independent causal stream.

To evaluate an existing enhanced-waveform directory with the seven supported speech-enhancement metrics:

```bash
python metrics/evaluate.py \
  --clean-dir "$DATA_ROOT/dns/clean" \
  --noisy-dir "$DATA_ROOT/dns/noisy" \
  --denoised-dir outputs/dns/crm/wav \
  --out-dir outputs/dns/crm/metrics \
  --method crm \
  --references data/manifests/dns.csv \
  --fs 16000
```

## Acknowledgements

We thank the authors of [SETTA](https://github.com/tobiaaa/SETTA) for releasing the pretrained CMGAN checkpoint and compatibility implementation, and the authors of [CMGAN](https://github.com/ruizhecao96/CMGAN) for the speech-enhancement backbone.

## Citation

If you find this work useful, please cite:

```bibtex
@misc{shan2026continual,
  title  = {Continual Residual Memory for Gradient-Free Test-Time Adaptation in Speech Enhancement},
  author = {Shan, Yijia and Wang, Tianrui and Wang, Zixiang and Wang, Yu and Chen, Xie},
  year   = {2026}
}
```

## License

Project code is released under [GPL-3.0](LICENSE). Third-party components and protocol metadata remain subject to their respective terms; see [`THIRD_PARTY.md`](THIRD_PARTY.md).
