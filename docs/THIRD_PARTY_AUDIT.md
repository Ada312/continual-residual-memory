# Third-Party Redistribution Audit

Audit date: 2026-09-15. No file was removed as part of this audit. Removal or
externalization requires an explicit maintainer decision and a replacement
that preserves the paper reproduction chain.

| Tracked path | Origin | Status | Recommended action |
| --- | --- | --- | --- |
| `crm/`, CRM configs, CRM checkpoints | This project | Self-authored/frozen project artifacts | Keep |
| `scripts/`, `data/*.py`, cross-backbone adapters | This project | Local orchestration and preparation code | Keep |
| `backbones/cmgan/`, `backbones/registry.py` | SETTA `08ea624f`, ultimately CMGAN; Conformer also traces to lucidrains | GPL-3.0 derivative plus MIT upstream notices; mix of unchanged and modified files | Prefer a pinned external SETTA/CMGAN dependency after an adapter is validated; until then retain with `THIRD_PARTY_NOTICES.md` because removal breaks main inference |
| `metrics/_frozen/` | SETTA `08ea624f`; STOI adapted from pystoi | GPL-3.0 derivative, with MIT pystoi notice | Retain for exact seven-metric reproducibility, with notices and modification disclosure |
| `patches/setta_final_baselines.patch` | Patch against SETTA `08ea624f` | GPL-3.0 derivative | Retain for exact LaDen/MPol reproduction, with GPL attribution |
| `checkpoints/baselines/WavLM_EARS_map.th` | Byte-identical SETTA tracked checkpoint | Third-party binary, 525,503 bytes | Externalize to the pinned SETTA download and keep SHA256/instructions; removal pending maintainer approval |
| `checkpoints/crm/**` | Independently trained CRM modules | Project weights; no upstream backbone tensors included | Keep |
| `manifests/dns.csv`, `ears_d.csv`, `musan_music*.csv` | Project-generated portable identities/orders derived from public datasets | No waveform; essential protocol metadata | Keep with dataset attribution and upstream terms |
| `manifests/training/metadata.txt`, `subset_manifest.json` | Project-generated EARS-WHAM selection metadata | No waveform; EARS benchmark declares CC BY-NC 4.0 | Keep only if exact training-subset reproducibility is required; otherwise regenerate externally. Mark as CC BY-NC-derived metadata |
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
5. Do not redistribute FlowSE source or weights unless its authors provide an
   explicit license or separate permission.

## Pending maintainer decisions

1. Approve externalizing `checkpoints/baselines/WavLM_EARS_map.th`.
2. Decide whether exact CMGAN compatibility justifies retaining the small
   vendored SETTA-derived implementation, or approve replacing it with a
   tested pinned-upstream adapter.
3. Confirm that EARS-WHAM selection metadata may remain under its explicit
   non-commercial dataset attribution.
