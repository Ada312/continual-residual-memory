# Checkpoints

## Main reproduction

| Component | Origin | Expected path | SHA256 | Redistribution |
| --- | --- | --- | --- | --- |
| Frozen CMGAN backbone | SETTA/LaDen-MPol, commit `08ea624f37dccc798f6bbffaf1f8f8e292c16e4b` | `checkpoints/external/cmgan_ears.th` | `2649bc63511f6c59bf2340f6ab2305c9c3f5fbe1aa949434c176ede3ec65108f` | External download |
| Utterance-local residual recovery | CRM authors | `checkpoints/crm/static_best.th` | `69c19970b7000b09d1b611b0dbc8786c24c9d9ca1c788cc077ff551225030523` | Included |
| Memory-conditioned refinement | CRM authors | `checkpoints/crm/dynamic_best.th` | `17d876b7a69d83b93dd335d3bad7c35bd4fda641f8176edb64094b974592be21` | Included |

The paper's CMGAN checkpoint is the file tracked by the official
[SETTA repository](https://github.com/tobiaaa/SETTA) at the pinned commit. The
SETTA repository is GPL-3.0, but no checkpoint-specific redistribution
statement was found. This repository therefore does not redistribute that
third-party binary and downloads it from its original location instead:

```bash
mkdir -p checkpoints/external
curl -L \
  https://raw.githubusercontent.com/tobiaaa/SETTA/08ea624f37dccc798f6bbffaf1f8f8e292c16e4b/checkpoints/cmgan_ears.th \
  -o checkpoints/external/cmgan_ears.th
echo "2649bc63511f6c59bf2340f6ab2305c9c3f5fbe1aa949434c176ede3ec65108f  checkpoints/external/cmgan_ears.th" \
  | sha256sum -c -
```

This checkpoint was released by SETTA and was not trained by the CRM authors.
Its architecture is CMGAN, originally released at
`https://github.com/ruizhecao96/CMGAN` under the MIT license. The embedded
compatibility implementation and its notices are documented in
[`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md).

## Optional experiments

The LaDen EARS map remains external; see
[`checkpoints/baselines/README.md`](baselines/README.md). Cross-backbone
checkpoint identities and acquisition instructions are isolated in
[`docs/CROSS_BACKBONE.md`](../docs/CROSS_BACKBONE.md).
