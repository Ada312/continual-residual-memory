# Efficiency Protocol

The official reference artifacts are under `results/efficiency/`. Report the actual measured scope rather than a theoretical full end-to-end FLOP count.

| Quantity | Frozen reference | Definition |
| --- | ---: | --- |
| CMGAN generator parameters | 1,834,833 | Source generator only |
| Utterance-local recovery parameters | 25,313 | Projector + gain head |
| Active refinement parameters | 11,776 | Basis and Coordinate Branches |
| Functional CRM overhead | 37,089 | Recovery + active refinement (excludes unused legacy adapter) |
| CMGAN / CRM RTF | 0.0339687589 / 0.0387355071 | RTX 4090, FP32, batch 1, ordered DNS; warmed repeated measurement |
| CMGAN / CRM counted compute | 104.589598 / 106.885930 GFLOPs/s audio | PyTorch `FlopCounterMode` registered operators only |
| Prototype storage | 129.75 KiB (130.75 KiB with candidate) | Theoretical fp32/int64 tensor storage at $K_{max}=64$, $F=257$ |

Run `scripts/count_parameters.py`, `scripts/estimate_memory_storage.py`, and `scripts/measure_rtf.py --help` to inspect the exact CLI. First produce a new RTF artifact with `scripts/measure_rtf.py` using the DNS manifest, CMGAN/CRM checkpoints, warm-up 10, repeats 3, K64 and `gamma_read=0.95`. Then pass that **new artifact** to `scripts/measure_flops.py --rtf-artifact ...` using the same files/device/flags. The profiler validates hashes, scope and configuration against the RTF run. The included frozen RTF JSON records the historical benchmark script hash and cannot be substituted into this renamed public command without regenerating it. Regeneration across hardware need not reproduce the frozen wall-clock RTF.

FFT/STFT/iSTFT and other unsupported operators are outside registered-op FLOPs; do not call the number full theoretical end-to-end FLOPs. Prototype storage is an analytical estimate, not measured peak GPU memory. No formal per-file inference latency or peak GPU memory claim is made.
