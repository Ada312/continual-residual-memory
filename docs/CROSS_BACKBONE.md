# Cross-Backbone Applicability

Each Source backbone is frozen. Its CRM recovery/refinement pair was independently trained and selected on EARS-WHAM (8,192 train / 632 held-out); all five are evaluated on the same frozen `manifests/dns.csv` (150, SHA256 `458df265...cff07b1`). CRM inference shares `configs/cmgan/final.json`, `scripts/infer_stream.py`, and the seven-metric evaluator. This is applicability of the **complete independently trained CRM pipeline**, not a universal backbone-independent checkpoint. No cross-backbone no-memory evaluation is claimed.

| Backbone / upstream commit | External Source checkpoint SHA256 | Included recovery / refinement SHA256 | Frozen Source inference setting |
| --- | --- | --- | --- |
| StoRM-50 `257e9636a7251ca40aa200753d5c0fe918e31879` | `bb160023f9b1c84eb6ef9ecff26ec02f8ebb96b2a53dfc9ded925d7acc862167` | `ca7d3814107f6ec1628888274ea84179c65a3763c3730beb368f6ef164e98961` / `8bfb9ee6b25254973ed83fae23e1f6eb90368ba59d48a8eb207edb98066dfef2` | 50 predictor steps, ALD, one corrector step, SNR 0.5 |
| GTCRN `502ebfab64da7c4a9af78dcb9c6ceef1ebb01c73` | `a0f0e04421d9fc1efe734d44191f5dc4b973ce59d6546af8d64ad9167e2a9944` | `263f6383c6810b34b714588b9463df8f01b3056c0120b03a13487474879397ee` / `a41b283f9eb79541048781875da7bfa486e7ff5ad73c21aafc8a9c48bcbb66b0` | VCTK-DEMAND STFT-mask wrapper |
| FastEnhancer-B `f85223bd546b27f39dc0744e0310dcd246f750a4` | `980ec00d9c3cb0497893c815c718a2fe44970329ae8477d22596d0a1373f2382` | `f7f413c0738eec20bc6589faca2f330f825a0a928e37a529203e3fc845f0c9c5` / `071f01f5e32c1c71a6a792176165e66742d415e23b2bc918e721e816852b4e5d` | Upstream B wrapper; pass `--backbone-config` |
| FlowSE (frozen artifact currently unresolved) | `9f4502703ccb9b252135884f1d72cb64bc6dbd23d0f23dc8b01c1b2f07bf329c` | `c97ac0af586e33234b40b74de72d7581f8dc3c6715a686dfa78269593806cef6` / `d664e19b436f37b9b060cf90b60bc74e54707dfe99d1d8a295e88db5794fdb4d` | Existing adapter: NFE 5, reverse 1.0 to 0.03 |
| UL-UNAS `00f7c700da43d38347f30a6ccebd86fcbc798e07` | `9a0656bf8e88d36865792d3065fed9bc33d5ab85e11ae3a1ae085b928fdb49a8` | `624f2437c6ab014344c2eaf413a25f633831b3a8bb6ecf85a068db89b294d011` / `14f5c7bb9461876d7173fd245e33ba02d4463b8b4e155ef7ddb8e45f0ec8a7c3` | Official DNS3 waveform inference |

Source adapters are under `backbones/cross_backbone/`; upstream repositories/commits are listed in `THIRD_PARTY.md`. For each model pass `--upstream-repo`, `--backbone-checkpoint`, DNS `noisy/` and `clean/` to `scripts/eval_cross_backbone.py`. For FastEnhancer-B also pass `--backbone-config`. After all five runs use `scripts/paired_bootstrap_cross_backbone.py --run OUTPUT_ROOT --iterations 20000 --seed 20260830` to regenerate paired PESQ CIs. Reference output CSVs are under `results/cross_backbone/`.

## FlowSE identity blocker

The paper citation supplied for public release is Wang et al., "FlowSE:
Efficient and High-Quality Speech Enhancement via Flow Matching," Interspeech
2025, DOI `10.21437/Interspeech.2025-1745`, with official repository
`https://github.com/Honee-W/FlowSE`. The intended external setup is:

```bash
git clone https://github.com/Honee-W/FlowSE.git .deps/FlowSE
git -C .deps/FlowSE checkout cfb81f171d804689bf7607afd0d12a6ee89de547
python -m pip install -r .deps/FlowSE/requirement.txt
mkdir -p checkpoints/external/flowse
curl -L \
  https://huggingface.co/flowse/wenetspeech4tts_Premium.pt.tar/resolve/df1bd3f8249a51f78d2b79bb037c9bdbcf660640/wenetspeech4tts_Premium.pt.tar \
  -o checkpoints/external/flowse/wenetspeech4tts_Premium.pt.tar
echo "c342175051d1ca38338993228e8a632181b9b556f9f1e310ac3e07bef54f446f  checkpoints/external/flowse/wenetspeech4tts_Premium.pt.tar" \
  | sha256sum -c -
```

That implementation uses a mel-spectrogram/DiT/vocoder pipeline. The current
`backbones/cross_backbone/apply_flowse_backbone.py` instead imports
`flowmse.model.VFModel` and an STFT-domain reverse-ODE solver from the distinct
`seongq/flowmse` project. Its frozen commit/checkpoint hashes and results cannot
be attributed to Wang et al. No valid Wang et al. cross-backbone evaluation
command is provided until a compatible adapter and corresponding CRM training
and DNS evaluation have been completed. This is a paper-result reproduction
blocker, not a licensing-only documentation issue.
