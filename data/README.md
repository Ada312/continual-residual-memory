# Dataset preparation

No dataset audio is redistributed. See [DATA.md](../docs/DATA.md) for the
official sources, expected counts, sample rates, frozen order and roles.
`prepare_ears_wham_from_benchmark.py` selects the frozen 8,192/632 protocol
from an official EARS-WHAM v1 build. `prepare_ears_wham_from_raw.py` is the
sample-equivalent sparse reconstruction path from raw EARS and WHAM. Both use
the frozen selection recorded in `manifests/training/subset_manifest.json`.
`prepare_ears_d.py` and `prepare_libri_musan.py` reconstruct the two generated
evaluation sets. The EARS-D entry defaults to the small frozen protocol files
under `manifests/protocol/` and reads audio metadata directly unless an
optional cache is supplied. Compare resulting identities and order against the
portable files under `manifests/` before evaluating.

Run `python data/<script>.py --help` for the supported input paths and
arguments. Never supply clean test speech to the causal memory inference.
