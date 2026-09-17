# Third-Party Resources

Third-party datasets and external model checkpoints are not redistributed in this repository. Obtain them from their official sources and follow their respective licenses and terms of use. Copyright and license notices for retained compatibility code are provided in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

## Source Code and Models

| Component | Official source | Repository use |
| --- | --- | --- |
| SETTA | [tobiaaa/SETTA](https://github.com/tobiaaa/SETTA) | Source of the pretrained CMGAN checkpoint and the compatibility implementation used by CRM |
| CMGAN | [ruizhecao96/CMGAN](https://github.com/ruizhecao96/CMGAN) | Frozen speech-enhancement backbone |
| Conformer | [lucidrains/conformer](https://github.com/lucidrains/conformer) | Component retained by the CMGAN implementation |
| pystoi | [mpariente/pystoi](https://github.com/mpariente/pystoi) | Basis for the retained STOI metric implementation |

The pretrained CMGAN checkpoint is downloaded from SETTA and placed at `checkpoints/external/cmgan_ears.th`. The checkpoint is not included in this repository.

## Datasets

| Dataset | Official source | Repository use |
| --- | --- | --- |
| EARS / EARS-WHAM | [EARS](https://github.com/facebookresearch/ears_dataset), [EARS benchmark](https://github.com/sp-uhh/ears_benchmark) | CRM training and held-out selection |
| WHAM! | [WHAM!](http://wham.whisper.ai/) | EARS-WHAM noise |
| DNS Challenge | [DNS Challenge](https://github.com/microsoft/DNS-Challenge) | DNS target evaluation |
| DEMAND | [DEMAND](https://doi.org/10.5281/zenodo.1227121) | EARS-D noise |
| LibriSpeech | [OpenSLR 12](https://www.openslr.org/12) | Libri-MUSAN clean speech |
| MUSAN | [OpenSLR 17](https://www.openslr.org/17) | Libri-MUSAN music noise |

No dataset waveform is tracked. Portable manifests contain only identities, order, and lightweight construction metadata. EARS-WHAM selection metadata under `manifests/training/` remains subject to the upstream CC BY-NC 4.0 terms rather than the project-code GPL-3.0 license.

## Python Packages

Python dependencies are installed from their normal package channels through `requirements.txt` and remain subject to their own licenses.
