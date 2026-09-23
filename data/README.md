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

Download the public 48 kHz EARS speech and WHAM noise releases. Arrange the raw
audio as follows (the WHAM command should point directly to its `audio/`
directory):

```text
/path/to/EARS/
├── p001/
│   └── *.wav
├── ...
└── p101/
    └── *.wav

/path/to/high_res_wham/audio/
└── *.wav
```

Reconstruct the exact fixed 8,192-example training split and 632-example
speaker-disjoint held-out split used in the paper:

```bash
export DATA_ROOT=/path/to/crm-data

python data/prepare_ears_wham_from_raw.py \
  --ears-dir /path/to/EARS \
  --wham-dir /path/to/high_res_wham/audio \
  --out-dir "$DATA_ROOT/ears_wham"
```

The selected identities are fixed in
`data/manifests/training/subset_manifest.json`. The script reads every speech
identity, noise identity, channel, offset, SNR, duration, and train/held-out
assignment directly from that manifest; it does not replay random dataset
generation or depend on filesystem enumeration order. It validates all
referenced files, 48 kHz sample rates, channels, and segment bounds before
writing output.

The paper dataset was created with Python's `[start:-1]` slicing for final EARS
segments whose recorded `speech_end` is `-1`. This omits the last waveform
sample. The reconstruction preserves that historical behavior deliberately for
sample-level compatibility with the data used in the paper.

If you already have a compatible EARS-WHAM v1 benchmark build containing its
original `train.csv` and `valid.csv`, the benchmark adapter remains available:

```bash
python data/prepare_ears_wham_from_benchmark.py \
  --ears-wham-root /path/to/EARS-WHAM-v1 \
  --out-dir "$DATA_ROOT/ears_wham"
```

## DNS

Download the official DNS 2020 synthetic test set from the
[DNS Challenge `interspeech2020/master` branch](https://github.com/microsoft/DNS-Challenge/tree/interspeech2020/master)
and use its `datasets/test_set/synthetic/no_reverb/{clean,noisy}` directories. The
official clean files are named `clean_fileid_<N>.wav`. Noisy files include a
descriptive prefix and end in `_fileid_<N>.wav`. CRM uses the shared name
`dns2020_no_reverb_fileid_<N>.wav` for each pair.

The preparation script matches clean and noisy files deterministically by their
file ID and creates symbolic links for exactly the 150 manifest entries without
copying the audio. Every required ID must have exactly one clean match and one
noisy match.

```bash
export DNS_RAW=/path/to/DNS-Challenge/datasets/test_set/synthetic/no_reverb

python data/prepare_dns.py \
  --dns-root "$DNS_RAW" \
  --output-root "$DATA_ROOT/dns"
```

The resulting filenames and causal evaluation order match
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
