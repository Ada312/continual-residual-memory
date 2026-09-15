# Third-Party Resources

Raw datasets and external backbone checkpoints are not redistributed. Obtain
them from the official sources below, verify the identities in
`docs/CHECKPOINTS.md` and `docs/CROSS_BACKBONE.md`, and follow each provider's
license and terms of use. Vendored compatibility code and derivative metadata
are inventoried in `THIRD_PARTY_NOTICES.md` and `docs/THIRD_PARTY_AUDIT.md`.

## Source Code and Models

| Component | Pinned upstream identity | License at audited revision | Repository use |
| --- | --- | --- | --- |
| CMGAN | `https://github.com/ruizhecao96/CMGAN.git`, `39946005d9b66fa1e824edf4bb6bc9a06088e443` | MIT | Main frozen Source architecture |
| SETTA (LaDen/MPol) | `https://github.com/tobiaaa/SETTA.git`, `08ea624f37dccc798f6bbffaf1f8f8e292c16e4b` | GPL-3.0 | Baseline patch and CMGAN/metric compatibility source |
| WavLM Large | `microsoft/wavlm-large`, revision `c1423ed94bb01d80a3f5ce5bc39f6026a0f4828c` | See model card | Frozen LaDen encoder |
| StoRM | `https://github.com/sp-uhh/storm.git`, `257e9636a7251ca40aa200753d5c0fe918e31879` | MIT | Frozen cross-backbone Source |
| GTCRN | `https://github.com/Xiaobin-Rong/gtcrn.git`, `502ebfab64da7c4a9af78dcb9c6ceef1ebb01c73` | MIT | Frozen cross-backbone Source |
| FastEnhancer | `https://github.com/aask1357/fastenhancer.git`, `f85223bd546b27f39dc0744e0310dcd246f750a4` | MIT | FastEnhancer-B frozen Source |
| FlowSE (Lee et al.) | `https://github.com/seongq/flowmse.git`, `f6b479d13fecc6cb6f12394f46dfc6799fb479b6` | **No license declared at the pinned revision** | Frozen flow-matching cross-backbone Source; upstream code and checkpoint remain external |
| UL-UNAS | `https://github.com/Xiaobin-Rong/ul-unas.git`, `00f7c700da43d38347f30a6ccebd86fcbc798e07` | MIT | Frozen cross-backbone Source |

The files under `backbones/cross_backbone/` are local adapters that import
separate upstream checkouts. They do not contain the five upstream model
implementations or weights.

## Datasets

| Dataset | Official source | Repository use |
| --- | --- | --- |
| EARS / EARS-WHAM | `https://github.com/facebookresearch/ears_dataset`, `https://github.com/sp-uhh/ears_benchmark` | Offline CRM training protocol; EARS benchmark declares CC BY-NC 4.0 |
| WHAM! | `http://wham.whisper.ai/` | EARS-WHAM noise |
| DNS Challenge | `https://github.com/microsoft/DNS-Challenge` | DNS 2020 synthetic no-reverb evaluation |
| DEMAND | `https://doi.org/10.5281/zenodo.1227121` | EARS-D noise |
| LibriSpeech | `https://www.openslr.org/12` | Libri-MUSAN clean speech |
| MUSAN | `https://www.openslr.org/17` | Libri-MUSAN music noise and carried-memory stream |

No waveform from these datasets is tracked. Portable manifests contain only
experiment identities/order and are addressed separately in the notices.
The EARS-WHAM selection files under `manifests/training/` remain subject to
the upstream CC BY-NC 4.0 terms, not the project-code GPL-3.0 license.

## Python Packages

Python packages are installed from their normal package channels and are not
vendored. The pinned runtime lists are `requirements-core.txt`,
`requirements-baselines.txt`, and `requirements-cross-backbone.txt`. Their own
licenses apply, including PyTorch/torchaudio, NumPy, SciPy, pandas, SoundFile,
PESQ, pystoi, torch-pesq, Hydra/OmegaConf, matplotlib, tqdm, einops, librosa,
and Transformers.
