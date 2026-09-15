# LaDen and MPol

The paper compares the same frozen CMGAN checkpoint on the same three ordered target streams. LaDen and MPol adapt a subset of CMGAN parameters using noisy test utterances and gradients; CRM changes only prototype state. They do **not** have matched adaptation compute. Output is generated before each baseline optimizer update. Clean target speech/labels are used only by the common external evaluator.

Start from `https://github.com/tobiaaa/SETTA.git` at commit `08ea624f37dccc798f6bbffaf1f8f8e292c16e4b`. Apply `patches/setta_final_baselines.patch` to this *separate checkout*. The patch updates exactly twelve formal-runtime source/config files relative to the pinned base; it has been tested by applying it in a fresh temporary checkout and byte-comparing every changed file with the frozen release candidate. `scripts/eval_baselines.py` rejects unpatched `run_da.py`, `adaptation/laden.py`, or `adaptation/mpol.py` by SHA256. The self-trained paper CRM implementation never imports SETTA at module import time.

- LaDen: 6,020 adapted normalization/output parameters; frozen `microsoft/wavlm-large` and externally downloaded EARS foundation map; AdamW `lr=5e-4` (PyTorch default weight decay 0.01); threshold 0.05; accumulation 32; consistency weight 0.01; power temperature 2.0; EMA 0.95. Only accepted utterances backpropagate. The final snapshot fixes WavLM to eval/local-only behavior without changing these method hyperparameters. Download the map from the pinned SETTA commit using `checkpoints/baselines/README.md`.
- MPol: same 6,020 parameters, AdamW `lr=5e-4`; `sign_weight=1.0`, Wasserstein weight 1.0; accumulation 8, clip 0.1, EMA 0.8 and power compression 0.3. Each utterance updates after producing output.

Example (from this repository root after preparing a separate SETTA checkout):

```bash
python scripts/eval_baselines.py --dataset dns --method laden \
  --setta-repo "$SETTA_REPO" --dataset-root "$DNS_ROOT" \
  --cmgan-checkpoint "$CMGAN_CHECKPOINT" --output-dir outputs/baselines
```

Run also for `mpol` and for datasets `ears_d` and `libri_musan`, using their corresponding dataset roots. Metrics use `metrics/evaluate.py`, the same evaluator as the CRM rows. See `docs/REPRODUCTION.md` for result aggregation.
