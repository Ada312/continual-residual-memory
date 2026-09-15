# Implementation Provenance

The paper-facing package was made by **mechanically extracting intact final class bodies** from the release-candidate frozen modules, without translating the mathematical operations. Attribute names used in final checkpoints were not renamed.

| Public class/body | Frozen source | Scope retained |
| --- | --- | --- |
| `crm/recovery.py` | `model/residual_projector/model.py` | `GatedResidualBlock`, `ResidualSpeechProjector`; historical `SignedResidualSpeechProjector` omitted |
| `crm/memory.py` | `model/residual_projector/prototype.py` | `PrototypeNoiseMemoryContext`, `CausalPrototypeResidualNoiseMemory`, `DynamicCausalPrototypeResidualNoiseMemory` |
| `crm/refinement.py` | same `prototype.py` | `PrototypeContinualResidualSpeechProjector` (readout and checkpoint parent), final `LowRankResidualDeltaPrototypeContinualResidualSpeechProjector` |
| `crm/_training/` | `scripts/train_residual_speech_projector.py`, `scripts/train_continual_residual_projector.py` | Final Stage 1 path retained; Stage 2 specialized to the paper architecture and active objective, with loss and update equivalence checked against the pre-cleanup implementation |
| `backbones/cmgan/`, `metrics/` | CMGAN Source support and frozen common seven-metric evaluator | Same checkpoint architecture, power-compressed Source transform and metric definitions; unused upstream dataset-loader helpers omitted from `transform_util.py` |

The public API is paper-facing (`UtteranceLocalRecovery`, `PrototypeMemory`, `MemoryConditionedRefinement`, `CRM`). `crm/_training/refinement.py` contains only the final Stage 2 architecture and objective: it initializes from the recovery checkpoint, freezes all parameters except `delta_basis` and `delta_coordinates`, and passes those parameters to AdamW with the recorded learning rate and weight decay. Historical architecture, router, checkpoint-loading and loss branches were removed. A fixed-seed synthetic batch comparison against the pre-cleanup compatibility implementation verified identical base and counterfactual losses and identical AdamW parameter updates for the final configuration.

Final checkpoint state-dict keys remain `input`, `blocks`, `output`, `delta_basis`, `delta_coordinates`, and the inactive `memory_adapter`. The final historical Dynamic runtime script's complete file hash was not saved; its executable loss/optimizer/command were recovered from content-addressed evidence and the final checkpoint. Consequently training can be run from these configs, but identical checkpoint reproduction from an unknown historical script snapshot is **not** claimed. Inference and all reference numerical results use the verified frozen checkpoints.

Training `K_train=2` with checkpoint `readout_mass=null` differs from deployment `K_max=64`, `gamma_read=0.95`. The latest paper now explicitly distinguishes these scopes. No algorithm change was made.
