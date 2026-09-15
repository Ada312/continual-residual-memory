"""Cheap checks of the frozen checkpoint and causal public interfaces."""

import csv
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import torch

from crm.model import CRM, load_recovery, load_refinement, new_memory
from scripts.infer_stream import load_memory, ordered_ids, save_memory, sha256


ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "configs/cmgan/final.json").read_text())


def test_paper_training_and_deployment_configs():
    recovery = json.loads((ROOT / "configs/cmgan/recovery_training.json").read_text())
    refinement = json.loads((ROOT / "configs/cmgan/refinement_training.json").read_text())
    assert (recovery["epochs"], recovery["learning_rate"], recovery["weight_decay"]) == (
        4, 3e-4, 1e-4
    )
    assert (refinement["epochs"], refinement["learning_rate"], refinement["weight_decay"]) == (
        3, 1e-3, 1e-4
    )
    assert refinement["k_train"] == 2
    assert refinement["readout_mass"] is None
    assert refinement["tau_cf"] == 0.02
    assert refinement["memory_dropout"] == 0.20
    assert (CONFIG["k_max"], CONFIG["tau_mem"], CONFIG["gamma_read"]) == (64, 0.5, 0.95)

    manifest = json.loads((ROOT / "manifests/training/subset_manifest.json").read_text())
    assert manifest["selected_train"]["pairs"] == 8192
    assert manifest["validation"]["pairs"] == 632
    assert manifest["speaker_overlap"] == []
    assert manifest["target_domain_data_used"] is False


def test_final_checkpoints_and_empty_memory():
    torch.set_num_threads(1)
    recovery = load_recovery(ROOT / "checkpoints/crm/static_best.th", CONFIG, torch.device("cpu"))
    refinement = load_refinement(ROOT / "checkpoints/crm/dynamic_best.th", CONFIG, torch.device("cpu"))
    crm = CRM(refinement, new_memory(CONFIG))
    noisy = torch.randn(4096) * 0.05
    backbone_estimate = noisy * 0.8
    with torch.inference_mode():
        no_memory, g_static, _ = recovery(noisy[None], backbone_estimate[None])
        enhanced, g_dyn, auxiliary = crm.infer(noisy, backbone_estimate)
    torch.testing.assert_close(enhanced, no_memory, rtol=0, atol=1e-6)
    torch.testing.assert_close(g_dyn, g_static, rtol=0, atol=1e-7)
    assert g_dyn.min() >= 0 and g_dyn.max() <= CONFIG["max_gain"]
    assert auxiliary["memory_delta"].abs().max() == 0
    assert crm.memory.step == 0

    crm.write(auxiliary)
    assert crm.memory.step == 1
    with torch.inference_mode():
        _, gain_next, auxiliary_next = crm.infer(noisy, backbone_estimate)
    assert gain_next.min() >= 0 and gain_next.max() <= CONFIG["max_gain"]
    assert auxiliary_next["memory_delta"].abs().max() <= CONFIG["alpha_dyn"]
    assert crm.memory.step == 1  # Inference itself must not write.


def test_refinement_training_parameter_selection():
    from crm.recovery import ResidualSpeechProjector
    from crm.refinement import MemoryConditionedRefinement

    config = json.loads((ROOT / "checkpoints/crm/config.json").read_text())
    recovery = ResidualSpeechProjector(
        channels=int(config["channels"]),
        max_gain=float(config["max_gain"]),
        initial_gain=float(config["initial_gain"]),
    )
    recovery.load_state_dict(torch.load(ROOT / "checkpoints/crm/static_best.th", weights_only=True))
    model = MemoryConditionedRefinement(
        channels=int(config["channels"]),
        max_gain=float(config["max_gain"]),
        initial_gain=float(config["initial_gain"]),
        memory_warmup=20,
        prototypes=2,
        posterior_temperature=0.5,
        reliability_power=0.0,
        posterior_confidence_power=0.0,
        readout_mass=None,
        delta_rank=4,
        memory_delta_gain=0.08,
    )
    model.initialize_from_static(recovery)
    model.requires_grad_(False)
    model.delta_basis.requires_grad_(True)
    model.delta_coordinates.requires_grad_(True)
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert trainable == {
        name for name, _ in model.named_parameters()
        if name.startswith(("delta_basis.", "delta_coordinates."))
    }
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 11776
    model.load_state_dict(torch.load(ROOT / "checkpoints/crm/dynamic_best.th", weights_only=True))


def test_refinement_context_builder_uses_public_memory_interface(monkeypatch):
    from crm._training import refinement

    metadata = pd.DataFrame(
        {
            "filename": ["p001_a.wav", "p001_b.wav"],
            "noise": ["wham_loc1_a.wav", "wham_loc1_b.wav"],
        }
    )
    waveform = torch.linspace(-0.1, 0.1, 4096)
    monkeypatch.setattr(refinement, "load_mono", lambda _: waveform)
    contexts = refinement.build_contexts(
        Path("noisy"),
        Path("source"),
        metadata,
        metadata["filename"].tolist(),
        noise_frame_fraction=0.3,
        memory_warmup=20,
        memory_prototypes=2,
        novelty_threshold=0.35,
        description="test memory",
    )
    assert set(contexts) == set(metadata["filename"])
    assert contexts["p001_a.wav"]["count"].sum() == 0
    assert contexts["p001_b.wav"]["count"].sum() == 1


def test_memory_roundtrip_and_checkpoint_guard(tmp_path):
    torch.set_num_threads(1)
    checkpoint = ROOT / "checkpoints/crm/dynamic_best.th"
    digest = sha256(checkpoint)
    model = load_refinement(checkpoint, CONFIG, torch.device("cpu"))
    first = CRM(model, new_memory(CONFIG))
    noisy = torch.randn(4096) * 0.05
    source = noisy * 0.8
    first.write(first.infer(noisy, source)[2])
    path = tmp_path / "memory.pt"
    save_memory(path, first, digest)
    second = CRM(model, new_memory(CONFIG))
    with pytest.raises(ValueError, match="checkpoint_sha256"):
        load_memory(path, second, "not-the-checkpoint")
    load_memory(path, second, digest)
    torch.testing.assert_close(first.infer(noisy, source)[0], second.infer(noisy, source)[0])
    assert first.memory.step == second.memory.step == 1


@pytest.mark.parametrize("manifest,count", [("dns.csv", 150), ("ears_d.csv", 886), ("musan_music.csv", 2620)])
def test_manifest_order_and_ids(manifest, count):
    path = ROOT / "manifests" / manifest
    ids = ordered_ids(path)
    assert len(ids) == count
    with path.open(newline="", encoding="utf-8") as handle:
        assert ids == [row["filename"] for row in csv.DictReader(handle)]


def test_manifest_rejects_duplicates_and_gaps(tmp_path):
    path = tmp_path / "manifest.csv"
    path.write_text("test_order,filename\n0,a.wav\n1,a.wav\n")
    with pytest.raises(ValueError, match="unique"):
        ordered_ids(path)
    path.write_text("test_order,filename\n0,a.wav\n2,b.wav\n")
    with pytest.raises(ValueError, match="contiguous"):
        ordered_ids(path)


def test_paired_bootstrap_small_input(tmp_path):
    columns = ["Filename", "PESQ", "STOI", "C_sig", "C_bak", "C_ovl", "SSNR", "SISDR"]
    static = pd.DataFrame([["a.wav", *([1.0] * 7)], ["b.wav", *([2.0] * 7)]], columns=columns)
    dynamic = static.copy()
    dynamic.loc[dynamic.Filename == "a.wav", columns[1:]] += 0.1
    dynamic.loc[dynamic.Filename == "b.wav", columns[1:]] += 0.3
    static.to_csv(tmp_path / "static.csv", index=False)
    dynamic.to_csv(tmp_path / "dynamic.csv", index=False)
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/paired_bootstrap_crm.py"),
         "--static", str(tmp_path / "static.csv"), "--dynamic", str(tmp_path / "dynamic.csv"),
         "--output", str(tmp_path / "bootstrap"), "--iterations", "200", "--seed", "20260830"],
        check=True, capture_output=True, text=True,
    )
    result = pd.read_csv(tmp_path / "bootstrap/paired_bootstrap_ci.csv")
    assert len(result) == 7
    assert result.utterances.eq(2).all()
    assert result.dynamic_minus_static_mean.between(0.199999, 0.200001).all()
    assert result.wins.eq(2).all()
