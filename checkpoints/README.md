# Checkpoints

The pretrained CMGAN checkpoint released by [SETTA](https://github.com/tobiaaa/SETTA)
and the CRM checkpoints are included in this repository:

| Component | Path |
| --- | --- |
| Frozen CMGAN backbone | `checkpoints/external/cmgan_ears.th` |
| Utterance-Local Residual Recovery | `checkpoints/crm/static_best.th` |
| Memory-Conditioned Refinement | `checkpoints/crm/dynamic_best.th` |

The external CMGAN checkpoint remains frozen during both CRM training stages and deployment.
