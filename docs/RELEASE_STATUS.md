# Release Verification Status

This is a new repository assembled from the frozen release candidate, not a
copy of its Git history. The final model checkpoints and three-domain result
references are included; upstream checkpoint weights and dataset audio must
be obtained separately. Nothing in this document asserts that full benchmarks
have been rerun in this checkout.

## Verified in this checkout

- `ruff check .` and
  `python -m compileall -q crm backbones data metrics scripts` succeed;
  all public `scripts/*.py`, Source, five backbone wrappers, dataset preparation
  entries, and the common metric entry accept `--help`.
- `python -m pytest -q tests` verifies strict final checkpoint loading,
  no-memory output and gain equality, bounded correction/final gain,
  inference-before-write, save/load continuity and manifest identities/order.
- CMGAN Source generated from one frozen DNS input with the official Source
  checkpoint matches the original frozen *inference script* output exactly.
  Compared with its archived DNS Source WAV, the maximum difference was
  0.000213623 (7 PCM16 steps), mean absolute difference 0.000009199; do not
  assert byte-for-byte identity to historical GPU output.
- `sha256sum -c docs/PAPER_DNS_FIGURE_SHA256SUMS`,
  `python scripts/plot_continual.py` and
  `python scripts/aggregate_paper_results.py --results-root results
  --output-dir outputs/check_tables` succeed using frozen reference inputs.
  `scripts/aggregate_cross_domain.py` recreates the manuscript's matched and
  Libri-MUSAN-to-DNS rows directly from the frozen 150-pair DNS artifact.
  `scripts/aggregate_cross_backbone.py` regenerates all five Source/CRM
  seven-metric rows from the frozen per-file result CSVs.
- Cross-backbone paired bootstrap, 20,000 draws and seed `20260830`, gives
  all five frozen mean deltas and 95% CI endpoints exactly from the included
  per-file inputs, writing only to an independent temporary directory.
- `scripts/count_parameters.py` reproduces the recorded 1,834,833 Source,
  25,313 recovery, 11,776 active refinement and 37,089 functional extra
  parameter counts. The full 39,137-element refinement checkpoint contains
  an inactive legacy adapter; its size is not treated as active refinement.
- Training and cross-backbone orchestration CLIs parse. Formal full-dataset
  training, cross-backbone inference, third-party baseline evaluation and
  profiler runs were **not** executed for this release-only verification.
- A clean-room check on 2026-09-15 used a fresh clone and a newly created
  Python 3.9 environment. The included recovery/refinement checkpoints ran an
  ordered three-utterance synthetic stream, preserved exact first-utterance
  no-memory equality, wrote and reloaded prototype state, and completed the
  common seven-metric WAV-to-CSV pipeline. A six-train/two-held-out synthetic
  Stage 2 smoke run built grouped causal contexts, completed one optimization
  epoch, saved a new checkpoint, and used that checkpoint for causal inference.
  This validates executable plumbing only; it is not a substitute for the
  formal 8,192/632 EARS-WHAM training experiment.

## Publication qualifications

1. The final historical Stage 2 script snapshot was not preserved. The
   reconstructed entry and confirmed optimizer/loss evidence are documented
   in `IMPLEMENTATION_PROVENANCE.md`; an *identical historical checkpoint
   retraining* guarantee would be unsupported.
2. The frozen RTF reference embeds the old benchmark script hash. The
   relocated public RTF script must create a fresh runtime artifact before
   the public FLOPs profiler will accept it. The frozen numerical reference
   is retained; no new benchmark has been substituted for it.
3. The Stage 2 trainer has been specialized to the final paper configuration.
   Its loss, counterfactual terms, gradients and one AdamW update were checked
   against the pre-cleanup implementation on the same fixed-seed batch.
4. FlowSE draws a Gaussian conditional prior during Source inference. Its
   archived formal command did not record a random seed. The repository
   preserves the exact model/checkpoint/protocol and frozen per-file result
   artifacts, but does not claim bit-identical regeneration of the historical
   FlowSE Source waveforms.

The CRM core and paper-result reproduction paths are suitable for public code
release under the stated third-party terms. The core inference API,
paper-facing training/evaluation entries, frozen result artifacts, and
aggregation/statistics paths are present and internally checked.
The historical Stage 2 snapshot limitation above remains a reproducibility
qualification: the released recipe is executable and its active computation is
validated, but byte-identical regeneration of the provided checkpoint is not
claimed. The manuscript correctly distinguishes training readout from
deployment readout.

The FlowSE cross-backbone row is attributed to the implementation actually
used: Lee et al.'s `seongq/flowmse` at commit
`f6b479d13fecc6cb6f12394f46dfc6799fb479b6`, with the official VoiceBank-DEMAND
checkpoint, NFE 5 and reverse ODE 1.0 to 0.03. The previous model-identity
publication blocker is therefore resolved.
