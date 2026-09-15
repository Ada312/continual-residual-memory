# External Baseline Artifacts

Third-party baseline artifacts are not redistributed. LaDen requires the EARS
foundation map published in the pinned SETTA repository:

```bash
mkdir -p checkpoints/baselines
curl -L \
  https://raw.githubusercontent.com/tobiaaa/SETTA/08ea624f37dccc798f6bbffaf1f8f8e292c16e4b/checkpoints/WavLM_EARS_map.th \
  -o checkpoints/baselines/WavLM_EARS_map.th
echo "3f2102adb72c406cd34ad212d76b19db56e53bec2658b1bfb861fda1a1708963  checkpoints/baselines/WavLM_EARS_map.th" \
  | sha256sum -c -
```

Expected path: `checkpoints/baselines/WavLM_EARS_map.th`.

The WavLM Large encoder is downloaded separately from
`microsoft/wavlm-large` at revision
`c1423ed94bb01d80a3f5ce5bc39f6026a0f4828c`; see `docs/CHECKPOINTS.md`.
