# Data Preparation

This repository does not redistribute dataset audio. Download each dataset from
its official source and follow its license and terms of use.

| Dataset | Use | Source |
| --- | --- | --- |
| EARS-WHAM | CRM training and held-out selection | [EARS](https://github.com/facebookresearch/ears_dataset), [EARS benchmark](https://github.com/sp-uhh/ears_benchmark), [WHAM!](http://wham.whisper.ai/) |
| DNS 2020 synthetic no-reverb | Target evaluation | [DNS Challenge](https://github.com/microsoft/DNS-Challenge) |
| EARS-D | Target evaluation | [EARS](https://github.com/facebookresearch/ears_dataset), [DEMAND](https://doi.org/10.5281/zenodo.1227121) |
| Libri-MUSAN | Target evaluation | [LibriSpeech](https://www.openslr.org/12), [MUSAN](https://www.openslr.org/17) |

## Directory Layout

Prepared datasets use matching `clean/` and `noisy/` filenames. The EARS-WHAM
training view additionally contains `metadata.txt`; frozen CMGAN estimates are
written to `source/` before CRM training.

```text
crm-data/
├── ears_wham/
│   ├── clean/
│   ├── noisy/
│   ├── source/
│   └── metadata.txt
├── dns/
│   ├── clean/
│   └── noisy/
├── ears_d/
│   ├── clean/
│   └── noisy/
└── libri_musan/
    ├── clean/
    └── noisy/
```

All audio is evaluated at 16 kHz. Clean target speech is used only for offline
evaluation, never for CRM inference or memory updates.

## EARS-WHAM

Prepare the fixed 8,192-example training split and 632-example speaker-disjoint
held-out split from an official EARS-WHAM v1 build:

```bash
python data/prepare_ears_wham_from_benchmark.py \
  --ears-wham-root /path/to/EARS-WHAM \
  --out-dir "$DATA_ROOT/ears_wham"
```

The selected identities are fixed in
`data/manifests/training/subset_manifest.json`. The preparation script consumes
that metadata directly. `data/prepare_ears_wham_from_raw.py` provides an
alternative reconstruction path from raw EARS and WHAM data.

## DNS

Place the DNS 2020 synthetic no-reverberation files under
`$DATA_ROOT/dns/{clean,noisy}`. Filenames and causal evaluation order must match
`data/manifests/dns.csv`.

## EARS-D

Construct EARS-D from the official EARS and DEMAND releases:

```bash
python data/prepare_ears_d.py \
  --ears-dir /path/to/EARS \
  --wham-dir /path/to/WHAM/audio \
  --test-files data/manifests/protocol/ears_benchmark_v1_test_files.json \
  --demand-index data/manifests/protocol/demand_16k_index.csv \
  --demand-dir /path/to/DEMAND/16k \
  --ears-w-out "$DATA_ROOT/ears_w_test" \
  --ears-d-out "$DATA_ROOT/ears_d"
```

The resulting 886 examples must follow `data/manifests/ears_d.csv`.

## Libri-MUSAN

Construct the 2,620-example LibriSpeech test-clean and MUSAN music evaluation
set:

```bash
python data/prepare_libri_musan.py \
  --manifest data/manifests/musan_music_construction.csv \
  --librispeech-root /path/to/LibriSpeech/test-clean \
  --musan-root /path/to/musan \
  --output-root "$DATA_ROOT/libri_musan"
```

The fixed evaluation order is stored in `data/manifests/musan_music.csv`.
