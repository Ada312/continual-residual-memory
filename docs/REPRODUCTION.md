# Reproducing the Paper

All commands run from the repository root. Set paths to legally obtained datasets and upstream checkpoints; never write clean target files into memory inputs. The small self-trained CRM checkpoints and frozen reference CSVs are included. `--help` on each entry lists optional flags; `--dry-run` on orchestration entries prints commands without training/inference.

## 1. Environment and EARS-WHAM

Create the pinned environment from `environment.yml` (or install `requirements-core.txt`). Download the external CMGAN checkpoint using `checkpoints/README.md`. Prepare EARS-WHAM using either the official benchmark output or the sample-equivalent raw-data path (8,192 training, 632 speaker-disjoint held-out):

```bash
python data/prepare_ears_wham_from_benchmark.py \
  --ears-wham-root "$EARS_WHAM_V1_ROOT" --out-dir "$EARS_ROOT"
# Alternative: python data/prepare_ears_wham_from_raw.py --help
```

Verify the generated `subset_manifest.json` against `manifests/training/subset_manifest.json`. Build the source cache:

```bash
python backbones/cmgan/inference.py --noisy-dir "$EARS_ROOT/noisy" \
  --output-dir outputs/ears_source \
  --checkpoint checkpoints/external/cmgan_ears.th
```

Train the two paper stages (frozen recipe, no target data):

```bash
python scripts/train_recovery.py --clean-dir "$EARS_ROOT/clean" \
  --noisy-dir "$EARS_ROOT/noisy" --source-dir outputs/ears_source \
  --output-dir outputs/train_recovery
python scripts/train_refinement.py --clean-dir "$EARS_ROOT/clean" \
  --noisy-dir "$EARS_ROOT/noisy" --source-dir outputs/ears_source \
  --metadata "$EARS_ROOT/metadata.txt" --recovery-checkpoint outputs/train_recovery/best.th \
  --context-cache outputs/context_k2_grouped.pt --output-dir outputs/train_refinement
```

Stage 2 reads `config.json` beside its Stage 1 checkpoint. `tau_cf=0.02` is fixed inside the retained training computation; no optimizer or memory hyperparameters were changed. Re-training can produce a different checkpoint across hardware/seeds. For a paper inference reproduction, use the **included final checkpoints**.

## 2. Three Target Streams and Baselines

For each target root containing `noisy/` and `clean/`, run with the correct `--dataset` and a **fresh empty prototype memory**:

```bash
python scripts/eval_main.py --dataset dns --noisy-dir "$DNS_ROOT/noisy" \
  --clean-dir "$DNS_ROOT/clean" --cmgan-checkpoint "$CMGAN_CHECKPOINT" \
  --output-dir outputs/main/dns
```

Repeat for `--dataset ears_d` into `outputs/main/ears_d` and `--dataset libri_musan` into `outputs/main/libri_musan`. Each call produces a frozen Backbone WAV, a CRM no-memory WAV, a CRM WAV and all seven metric CSVs, matched to the corresponding ordered manifest. Obtain the pinned SETTA checkout, apply the tested patch, then run `scripts/eval_baselines.py` for LaDen and MPol on each dataset; see `BASELINES.md`. All three methods begin from the same CMGAN checkpoint. Assemble and generate paper table rows:

```bash
python scripts/collect_results.py --main-root outputs/main \
  --baseline-root outputs/baselines --output-root outputs/paper_results
python scripts/aggregate_paper_results.py --results-root outputs/paper_results \
  --output-dir outputs/tables
```

For CI on a single dataset pass its Static/CRM per-utterance CSVs to `scripts/paired_bootstrap_crm.py --static ... --dynamic ... --output outputs/ci/dns --iterations 20000 --seed 20260830`. The included frozen reference CSVs under `results/main/per_utterance/` can generate the three-domain table and Figure immediately, without any audio processing.

## 3. Continual Progression and Cross-Domain Transition

```bash
sha256sum -c docs/PAPER_DNS_FIGURE_SHA256SUMS
python scripts/plot_continual.py
```

`outputs/figures/{pesq,covl}.png` shows Dynamic/CRM minus no-memory recovery under a causal trailing 10-utterance mean (no confidence bands). The script checks frozen DNS ID order and only consumes three frozen CSVs. `results/continual/` contains the supplied reference PNGs and hash metadata.

For cross-domain transition, first supply **complete, frozen** Libri-MUSAN Source WAVs and DNS Source WAVs, then run:

```bash
python scripts/eval_cross_domain.py \
  --musan-noisy-dir "$MUSAN_ROOT/noisy" --musan-source-dir "$MUSAN_SOURCE" \
  --dns-noisy-dir "$DNS_ROOT/noisy" --dns-source-dir "$DNS_SOURCE" \
  --dns-clean-dir "$DNS_ROOT/clean" --output-dir outputs/transition
```

This writes after each of all 2,620 Libri-MUSAN outputs, carries its final state to DNS without reset, then writes DNS only. Compare resulting DNS seven-metric rows with standard matched DNS CRM; included reference summaries are in `results/transition/`. No parameter training occurs on Libri-MUSAN.

Regenerate the manuscript's transition table directly from frozen paired DNS
metrics, or supply `--switch-csv outputs/transition/dns/metrics/crm_cross_domain_per_file_metrics.csv`
after a new full transition run. The latter must contain the same 150 DNS IDs:

```bash
python scripts/aggregate_cross_domain.py --output outputs/tables/cross_domain.csv
```

## 4. Five Backbones and Efficiency

For each pinned upstream checkout, pass its Source checkpoint, optional FastEnhancer config, DNS root and output root to `scripts/eval_cross_backbone.py --help`. The StoRM adapter enforces StoRM-50. Once all five output directories exist:

```bash
python scripts/paired_bootstrap_cross_backbone.py --run outputs/cross_backbone \
  --output-dir outputs/cross_backbone_ci --iterations 20000 --seed 20260830
```

For a statistics-only check on the included final per-file CSVs, substitute
`--run results/cross_backbone`; always specify a separate `--output-dir` so
the frozen bootstrap artifacts are not overwritten.

See `CROSS_BACKBONE.md` for commits/checkpoint identities and `EFFICIENCY.md` for parameters/RTF/FLOPs/storage commands and scope. The same `scripts/infer_stream.py` implements causal deployment for all six frozen backbones.
Generate the seven-metric cross-backbone table from the evaluated 150-per-file
CSVs with `python scripts/aggregate_cross_backbone.py --run
outputs/cross_backbone --output outputs/tables/cross_backbone.csv` (or omit
`--run` to regenerate from the included frozen reference inputs).
