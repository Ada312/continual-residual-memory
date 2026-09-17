# Data Protocol

No raw speech/noise, training cache, enhanced WAV, or clean reference is included. All paths below are supplied by the user; portable CSV manifests contain **relative IDs and frozen order**, not personal cluster paths. `test_order` is zero-based and stable; `scripts/infer_stream.py` reads it before any memory write.

| Data | Role | N | Sampling | Files / preparation |
| --- | --- | ---: | --- | --- |
| EARS-WHAM | Only offline training source | 8,192 train; 632 speaker-disjoint held-out | 16 kHz, 2-s random training crops | `data/prepare_ears_wham_from_benchmark.py` or `data/prepare_ears_wham_from_raw.py`; `manifests/training/` |
| DNS 2020 synthetic no-reverb | Independent target stream | 150 | 16 kHz | `manifests/dns.csv`; `noisy/` and `clean/` |
| EARS-D (DEMAND noise) | Independent target stream | 886 | 16 kHz | `data/prepare_ears_d.py`; `manifests/protocol/`; `manifests/ears_d.csv` |
| LibriSpeech test-clean + MUSAN music | Independent target stream and old-memory stage | 2,620 | 16 kHz, SNR in {-5,0,5,10} dB | `data/prepare_libri_musan.py`, `manifests/musan_music_construction.csv`, `manifests/musan_music.csv` |

Obtain each upstream dataset under its own license; invoke the preparation entry's `--help` for source roots and output naming. The training subset metadata lists the 8,824 train/held-out identities. `manifests/protocol/` contains the content-addressed EARS benchmark test selection and deterministic DEMAND 16 kHz segment index needed to reconstruct EARS-D; it contains no audio. Create the frozen Source waveform cache using `backbones/cmgan/inference.py` with the main CMGAN checkpoint, then point both training stages at `clean/`, `noisy/`, and `source/`. For each target stream, supply its `noisy/`, `clean/`, and Source directory to evaluation commands.

The manuscript's EARS-D frozen evaluation identity is its manifest; a historical upstream indexing identity beyond this list is **not** claimed. The held-out EARS-WHAM set is speaker-disjoint, not a strictly noise-disjoint new domain. Target-domain clean speech is used for offline metric computation only, not memory adaptation or checkpoint selection.
