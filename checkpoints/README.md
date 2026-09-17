# Checkpoints

Download the pretrained [`cmgan_ears.th`](https://raw.githubusercontent.com/tobiaaa/SETTA/main/checkpoints/cmgan_ears.th) checkpoint from [SETTA](https://github.com/tobiaaa/SETTA) and place it at:

```text
checkpoints/external/cmgan_ears.th
```

The CRM checkpoints are included in this repository:

| Stage | Path |
| --- | --- |
| Utterance-Local Residual Recovery | `checkpoints/crm/static_best.th` |
| Memory-Conditioned Refinement | `checkpoints/crm/dynamic_best.th` |

The external CMGAN checkpoint remains frozen during both CRM training stages and deployment.
