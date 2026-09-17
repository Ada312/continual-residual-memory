# Third-Party Notices

This file covers third-party material currently present in the repository.
Third-party datasets, external model implementations, and external backbone
weights are otherwise not redistributed. The project-level `LICENSE` is
GPL-3.0 because the release contains SETTA-derived GPL-3.0 material.

## SETTA-derived compatibility code

Source: `https://github.com/tobiaaa/SETTA`, commit
`08ea624f37dccc798f6bbffaf1f8f8e292c16e4b`.

Covered paths:

- `backbones/registry.py`
- `backbones/cmgan/`
- `metrics/_frozen/`
- `patches/setta_final_baselines.patch`

The CMGAN model/transforms and frozen evaluator were extracted from the pinned
SETTA implementation. Some files are unchanged; others were reduced, relocated,
or modified for strict checkpoint loading, standalone inference, current
PyTorch compatibility, robust metric failure handling, or removal of unused
training helpers. The baseline patch records this project's modifications to a
separate SETTA checkout. These modified versions were prepared in 2026.

SETTA is licensed under GNU GPL version 3. A complete copy is provided in
`LICENSE`. Copyright remains with the respective SETTA contributors.

## CMGAN

The SETTA-derived files under `backbones/cmgan/` implement CMGAN, originally
published at `https://github.com/ruizhecao96/CMGAN` and audited against commit
`39946005d9b66fa1e824edf4bb6bc9a06088e443`.

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

## Conformer implementation

`backbones/cmgan/modules.py` retains a Conformer implementation attributed by
SETTA/CMGAN to `https://github.com/lucidrains/conformer`.

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

## pystoi-derived metric code

`metrics/_frozen/stoi.py` is adapted from
`https://github.com/mpariente/pystoi` and is retained to preserve the frozen
evaluation behavior.

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

## WavLM EARS map

The LaDen EARS foundation map is not redistributed. It is downloaded from the
pinned SETTA repository into `checkpoints/baselines/WavLM_EARS_map.th` and
verified against SHA256
`3f2102adb72c406cd34ad212d76b19db56e53bec2658b1bfb861fda1a1708963`.
See `checkpoints/baselines/README.md`.

## SETTA CMGAN checkpoint

The main third-party CMGAN checkpoint is not redistributed. The exact
`checkpoints/cmgan_ears.th` file is obtained from SETTA commit
`08ea624f37dccc798f6bbffaf1f8f8e292c16e4b`, placed at
`checkpoints/external/cmgan_ears.th`, and verified against SHA256
`2649bc63511f6c59bf2340f6ab2305c9c3f5fbe1aa949434c176ede3ec65108f`.
The upstream repository is GPL-3.0, but no checkpoint-specific redistribution
statement was found; `checkpoints/README.md` therefore uses an upstream
download instead of bundling the binary.

## Dataset-derived metadata

Files under `manifests/` contain no audio. They record portable experiment
identity, construction metadata, and causal order. EARS/EARS-WHAM-derived
metadata, including `manifests/protocol/ears_benchmark_v1_test_files.json`,
is attributed to the EARS authors and should be treated under the upstream
CC BY-NC 4.0 terms. `manifests/protocol/demand_16k_index.csv` is deterministic
protocol metadata derived from the SETTA EARS-D indexing procedure and DEMAND;
their respective terms continue to apply. DNS, WHAM!, DEMAND, LibriSpeech, and
MUSAN retain their respective upstream terms. Files under `results/` are
numerical outputs produced by this project, not redistributed source audio.

## Non-vendored upstream projects

StoRM, FlowSE, GTCRN, FastEnhancer-B, and UL-UNAS implementations and Source
weights are not present. Only local invocation adapters and independently
trained CRM recovery/refinement weights are tracked. The FlowSE experiment
uses Lee et al., "FlowSE: Flow Matching-based Speech Enhancement"
(`https://github.com/seongq/flowmse`) at commit
`f6b479d13fecc6cb6f12394f46dfc6799fb479b6`. That pinned revision contains no
license file or license statement, so no redistribution permission is
inferred; its source and checkpoint remain external.
