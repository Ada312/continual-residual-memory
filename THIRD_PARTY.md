# Third-Party Resources and Notices

Third-party datasets are not redistributed in this repository. Obtain them
from their official sources and follow their respective licenses and terms of
use. The retained compatibility code and pretrained backbone remain subject to
the notices below.

## Resources

| Component | Official source | Repository use |
| --- | --- | --- |
| SETTA | [tobiaaa/SETTA](https://github.com/tobiaaa/SETTA) | Source of the pretrained CMGAN checkpoint and compatibility implementation |
| CMGAN | [ruizhecao96/CMGAN](https://github.com/ruizhecao96/CMGAN) | Frozen speech-enhancement backbone |
| Conformer | [lucidrains/conformer](https://github.com/lucidrains/conformer) | Component retained by the CMGAN implementation |
| pystoi | [mpariente/pystoi](https://github.com/mpariente/pystoi) | Basis for the retained STOI metric implementation |

The SETTA CMGAN checkpoint is included at
`checkpoints/external/cmgan_ears.th`. Copyright and third-party terms remain
with its upstream authors.

## Datasets

| Dataset | Official source | Repository use |
| --- | --- | --- |
| EARS / EARS-WHAM | [EARS](https://github.com/facebookresearch/ears_dataset), [EARS benchmark](https://github.com/sp-uhh/ears_benchmark) | CRM training and held-out selection |
| WHAM! | [WHAM!](http://wham.whisper.ai/) | EARS-WHAM noise |
| DNS Challenge | [DNS Challenge](https://github.com/microsoft/DNS-Challenge) | DNS target evaluation |
| DEMAND | [DEMAND](https://doi.org/10.5281/zenodo.1227121) | EARS-D noise |
| LibriSpeech | [OpenSLR 12](https://www.openslr.org/12) | Libri-MUSAN clean speech |
| MUSAN | [OpenSLR 17](https://www.openslr.org/17) | Libri-MUSAN music noise |

No dataset waveform is tracked. Portable metadata under `data/manifests/`
contains only identities, causal order, and lightweight construction records.
EARS/EARS-WHAM-derived metadata under `data/manifests/training/` remains
subject to the upstream CC BY-NC 4.0 terms rather than the project-code
GPL-3.0 license.

## SETTA-Derived Compatibility Code

Source: [tobiaaa/SETTA](https://github.com/tobiaaa/SETTA).

Covered paths:

- `src/backbones/registry.py`
- `src/backbones/cmgan/`
- `metrics/_frozen/`

The CMGAN model/transforms and frozen evaluator were extracted from SETTA.
Some files are unchanged; others were reduced, relocated, or modified for
strict checkpoint loading, standalone inference, current PyTorch
compatibility, robust metric failure handling, or removal of unused training
helpers. These modified versions were prepared in 2026.

SETTA is licensed under GNU GPL version 3. A complete copy is provided in
`LICENSE`. Copyright remains with the respective SETTA contributors.

## CMGAN

The SETTA-derived files under `src/backbones/cmgan/` implement CMGAN, originally
published at [ruizhecao96/CMGAN](https://github.com/ruizhecao96/CMGAN).

MIT License

Copyright (c) 2022 Ruizhe Cao

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Conformer Implementation

`src/backbones/cmgan/modules.py` retains a Conformer implementation attributed by
SETTA/CMGAN to [lucidrains/conformer](https://github.com/lucidrains/conformer).

MIT License

Copyright (c) 2020 Phil Wang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## pystoi-Derived Metric Code

`metrics/_frozen/stoi.py` is adapted from
[mpariente/pystoi](https://github.com/mpariente/pystoi) and is retained to
preserve the frozen evaluation behavior.

MIT License

Copyright (c) 2018 Pariente Manuel

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Dataset-Derived Metadata

Files under `data/manifests/` contain no audio. EARS/EARS-WHAM-derived
metadata is attributed to the EARS authors and should be treated under the
upstream CC BY-NC 4.0 terms. DEMAND protocol metadata is derived from the
SETTA EARS-D indexing procedure and DEMAND; their respective terms continue
to apply. DNS, WHAM!, DEMAND, LibriSpeech, and MUSAN retain their respective
upstream terms.

## Python Packages

Python dependencies installed through `requirements.txt` remain subject to
their own licenses.
