# Third-Party Redistribution Audit

Audit date: 2026-09-15. This audit records both retained files and external
resources referenced by the reproduction workflow. The maintainer approved
externalizing the WavLM EARS map; its tracked copy was removed and replaced by
a pinned download plus SHA256 verification. No other audited component was
removed or externalized.

| Repository or expected path | Origin | Status | Action |
| --- | --- | --- | --- |
| `crm/`, CRM configs, CRM checkpoints | This project | Self-authored/frozen project artifacts | Keep |
| `scripts/`, `data/*.py`, cross-backbone adapters | This project | Local orchestration and preparation code | Keep |
| `backbones/cmgan/`, `backbones/registry.py` | SETTA `08ea624f`, ultimately CMGAN; Conformer also traces to lucidrains | GPL-3.0 derivative plus MIT upstream notices; mix of unchanged and modified files | **Keep as approved** with `THIRD_PARTY_NOTICES.md`; this is the clean-room-tested main-backbone implementation |
| `metrics/_frozen/` | SETTA `08ea624f`; STOI adapted from pystoi | GPL-3.0 derivative, with MIT pystoi notice | Retain for exact seven-metric reproducibility, with notices and modification disclosure |
| `patches/setta_final_baselines.patch` | Patch against SETTA `08ea624f` | GPL-3.0 derivative | Retain for exact LaDen/MPol reproduction, with GPL attribution |
| `checkpoints/external/cmgan_ears.th` | SETTA `08ea624f` tracked CMGAN checkpoint | Upstream repository is GPL-3.0; no checkpoint-specific redistribution statement found | **Externalized**; download from pinned upstream URL and verify SHA256 |
| `checkpoints/baselines/WavLM_EARS_map.th` | Byte-identical SETTA tracked checkpoint | Third-party binary, 525,503 bytes | **Externalized**; ignored locally and restored from pinned SETTA using documented SHA256 |
| `checkpoints/crm/**` | Independently trained CRM modules | Project weights; no upstream backbone tensors included | Keep |
| `manifests/dns.csv`, `ears_d.csv`, `musan_music*.csv` | Project-generated portable identities/orders derived from public datasets | No waveform; essential protocol metadata | Keep with dataset attribution and upstream terms |
| `manifests/training/metadata.txt`, `subset_manifest.json` | Project-generated EARS-WHAM selection metadata | No waveform; EARS benchmark declares CC BY-NC 4.0 | **Keep as approved** for exact training-subset reproducibility, under the separate scope in `manifests/training/README.md` |
| `manifests/protocol/` | EARS benchmark test selection and SETTA/DEMAND indexing metadata | No waveform; upstream dataset/code terms apply | Keep for deterministic EARS-D reconstruction with explicit attribution |
| `results/**` | Project-computed metrics, statistics, and figures | No source audio or external checkpoint | Keep as compact paper regression references |

## File-level provenance findings

Comparison against SETTA commit `08ea624f37dccc798f6bbffaf1f8f8e292c16e4b`
found exact copies at audit time for CMGAN discriminator/modules/utils and
several transform/evaluator files. CMGAN generator/model/transforms and frozen
PESQ/composite files contain local changes. `backbones/cmgan/transform_util.py`
is a reduced extraction of SETTA `datasets/util.py`. Modification scope is
summarized in `THIRD_PARTY_NOTICES.md` and `docs/IMPLEMENTATION_PROVENANCE.md`.

The five cross-backbone adapters do not embed StoRM, FlowSE, GTCRN,
FastEnhancer-B, or UL-UNAS model source. Their upstream commits must be checked
out separately. The corresponding CRM checkpoints contain only this project's
recovery/refinement modules.

## Dependency policy

1. Do not add raw dataset audio, external backbone source trees, or external
   backbone checkpoints.
2. Prefer official download instructions plus pinned commits and SHA256 values.
3. Keep vendored code only when it is required for exact behavior and its
   redistribution terms are satisfied.
4. Record copyright, license, upstream identity, and modifications for every
   retained vendored component.
5. Do not redistribute Lee et al. FlowSE source or weights unless its authors
   provide an explicit license or separate permission.

## Maintainer decisions

Resolved decisions: the WavLM EARS map is external; the tested embedded CMGAN
compatibility implementation remains with full notices; EARS-WHAM selection
metadata remains with explicit CC BY-NC 4.0 scope; FlowSE is the actually
evaluated Lee et al. `seongq/flowmse` implementation and remains external.
