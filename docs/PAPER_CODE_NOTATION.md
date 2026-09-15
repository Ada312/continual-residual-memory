# Paper-to-Code Notation

The latest manuscript defines the notation. Local Python names are descriptive; aliases retain original checkpoint keys. Utterance index $t$, TF frame index $\tau$, and frequency index $f$ are distinct.

| Paper symbol | Paper meaning | Public implementation / preferred name |
| --- | --- | --- |
| $y_t$ | Current noisy waveform | `noisy_waveform` in `crm/residual.py`, `crm/model.py`, `scripts/infer_stream.py` |
| $G_\theta$, $\hat x_t$ | Frozen backbone, waveform estimate | `backbones/cmgan/inference.py`; `backbone_estimate` |
| $r_t=y_t-\hat x_t$ | Discarded waveform residual | `residual` in `crm/residual.py`; `noisy - source` in preserved model forwards |
| $\hat X_t$, $R_t$ | STFTs of backbone estimate and residual | `x_hat_stft`, `residual_stft`; checkpoint-compatible locals `source_spectrum`, `residual_spectrum` |
| $\Phi_t$ | Seven-channel TF features $[B,7,F,T]$ | `tf_features`; `ResidualSpeechProjector.features()` in `crm/recovery.py` |
| $P_s$, $H_t$ | Utterance-local projector and hidden tensor $[B,32,F,T]$ | `UtteranceLocalRecovery`, `hidden`; checkpoint keys `input`, `blocks` |
| $\ell_t$, $g_t^{\mathrm{static}}$ | Local logit and bounded TF gain | `static_logit`, `static_gain` / `gain`; checkpoint key `output` |
| $\hat X_t^{\mathrm{static}}$ | No-memory enhanced spectrum | `estimate_spectrum` in `crm/recovery.py` |
| $M_{t-1}$ | Prototype state before utterance $t$ | `memory_prev = memory.context()` in `crm/model.py` |
| $u_t$ | Full-utterance residual log-power median query | `observation` in `crm/refinement.py::_memory_features()` |
| $\mu_m,v_m,n_m$ | Prototype statistics | `memory.means`, `variances`, `counts` in `crm/memory.py` |
| $Z_t$ | Six memory-context TF maps | `context_features` from `memory_features()` in `crm/refinement.py` |
| $B_t=\{b_{t,k}\}$ | Current-utterance basis maps | `basis = tanh(self.delta_basis(hidden.detach()))`, $[B,d,F,T]$ |
| $C_t=\{c_{t,k}\}$ | Memory-conditioned coordinate maps | `coordinates = tanh(self.delta_coordinates(context_features))` |
| $\Delta g_t$ | Signed low-rank correction | `delta_gain` (paper-facing `delta_g`); `alpha_dyn=memory_delta_gain` |
| $g_t^{\mathrm{dyn}}$ | Final gain bounded by $[0,g_{\max}]$ | `gain = (static_gain + delta_gain).clamp(...)` |
| $\hat X_t^{\mathrm{dyn}}$ | Final enhanced spectrum | `estimate_spectrum` in final low-rank forward |
| $q_{t,\tau}$ | Backbone-output occupancy per frame | `source_fraction` in `crm/memory.py::_observation()` |
| $o_t$ | Lowest-occupancy-30% residual observation | `_observation()` output, passed to prototype `update()` |
| $M_t$ | Post-inference updated prototype state | `CRM.write()` after `torchaudio.save()` in `scripts/infer_stream.py` |

`crm/refinement.py` retains the historical Python class names `PrototypeContinualResidualSpeechProjector` and `LowRankResidualDeltaPrototypeContinualResidualSpeechProjector` for frozen training imports. The checkpoint depends on attribute keys, not Python class names: renaming `self.input`, `self.blocks`, `self.output`, `self.delta_basis`, `self.delta_coordinates`, or the inactive `self.memory_adapter` would break final `state_dict` loading. Public aliases `UtteranceLocalRecovery`, `PrototypeMemory`, and `MemoryConditionedRefinement` follow paper terminology. `memory_adapter` is a loaded **inactive** legacy tensor and is not counted as active refinement parameters.

The training entry uses `K_train=2`, with no mass truncation in its frozen checkpoint configuration. Deployment uses `K_max=64` and `gamma_read=0.95`, as the current manuscript now states. No training or deployment behavior was changed.
