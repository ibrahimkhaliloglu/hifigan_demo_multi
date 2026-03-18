#!/usr/bin/env python3
"""
pt_to_nemo.py
-------------
Convert a NVIDIA DeepLearningExamples FastPitch .pt checkpoint
to a portable .nemo file.

Usage:
    python pt_to_nemo.py --input /path/to/model.pt
    python pt_to_nemo.py --input /path/to/model.pt --output /path/to/model.nemo
"""

import argparse
import io
import os
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path

import torch
from omegaconf import OmegaConf


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Convert DLE FastPitch .pt → .nemo")
    p.add_argument("--input",  "-i", required=True,  help="Path to input .pt file")
    p.add_argument("--output", "-o", required=False, help="Path to output .nemo file "
                                                          "(default: same dir/name as input)")
    return p.parse_args()


# ── Load .pt and extract config + state_dict ──────────────────────────────────
def load_pt(pt_path: str) -> tuple:
    print(f"[1/5] Loading checkpoint: {pt_path}")
    size_mb = os.path.getsize(pt_path) / 1e6
    print(f"      Size: {size_mb:.1f} MB")

    ckpt = torch.load(pt_path, map_location="cpu")

    if "config" not in ckpt:
        raise KeyError(
            "Key 'config' not found in checkpoint. "
            "Expected a DLE FastPitch checkpoint with keys: config, state_dict."
        )
    if "state_dict" not in ckpt:
        raise KeyError("Key 'state_dict' not found in checkpoint.")

    cfg = ckpt["config"]
    print(f"      Speakers : {cfg['n_speakers']}")
    print(f"      Symbols  : {cfg['n_symbols']}")
    print(f"      Mel bins : {cfg['n_mel_channels']}")
    return ckpt, cfg


# ── Validate all required config keys are present ────────────────────────────
REQUIRED_KEYS = [
    "n_mel_channels", "n_symbols", "padding_idx", "n_speakers",
    "symbols_embedding_dim", "speaker_emb_weight",
    "in_fft_n_layers", "in_fft_n_heads", "in_fft_d_head",
    "in_fft_conv1d_kernel_size", "in_fft_conv1d_filter_size", "in_fft_output_size",
    "p_in_fft_dropout", "p_in_fft_dropatt", "p_in_fft_dropemb",
    "out_fft_n_layers", "out_fft_n_heads", "out_fft_d_head",
    "out_fft_conv1d_kernel_size", "out_fft_conv1d_filter_size", "out_fft_output_size",
    "p_out_fft_dropout", "p_out_fft_dropatt", "p_out_fft_dropemb",
    "dur_predictor_kernel_size", "dur_predictor_filter_size",
    "p_dur_predictor_dropout", "dur_predictor_n_layers",
    "pitch_predictor_kernel_size", "pitch_predictor_filter_size",
    "p_pitch_predictor_dropout", "pitch_predictor_n_layers",
    "pitch_embedding_kernel_size",
    "energy_predictor_kernel_size", "energy_predictor_filter_size",
    "p_energy_predictor_dropout", "energy_predictor_n_layers",
    "energy_conditioning", "energy_embedding_kernel_size",
]

def validate_config(cfg: dict):
    print("[2/5] Validating config keys...")
    missing = [k for k in REQUIRED_KEYS if k not in cfg]
    if missing:
        raise KeyError(
            f"The following keys are missing from the checkpoint config:\n"
            + "\n".join(f"  - {k}" for k in missing)
        )
    print(f"      All {len(REQUIRED_KEYS)} required keys present.")


# ── Build OmegaConf nemo_cfg from the DLE config dict ────────────────────────
def build_nemo_cfg(cfg: dict) -> OmegaConf:
    print("[3/5] Building .nemo config...")
    nemo_cfg = OmegaConf.create({
        # ── Top-level ────────────────────────────────────────────
        "sample_rate"           : 22050,
        "n_mel_channels"        : cfg["n_mel_channels"],
        "n_symbols"             : cfg["n_symbols"],
        "padding_idx"           : cfg["padding_idx"],
        "n_speakers"            : cfg["n_speakers"],
        "symbols_embedding_dim" : cfg["symbols_embedding_dim"],
        "speaker_emb_weight"    : cfg["speaker_emb_weight"],

        # ── Encoder (in_fft) ─────────────────────────────────────
        "input_fft": {
            "n_layers"      : cfg["in_fft_n_layers"],
            "n_head"        : cfg["in_fft_n_heads"],
            "d_head"        : cfg["in_fft_d_head"],
            "d_inner"       : cfg["in_fft_conv1d_filter_size"],
            "kernel_size"   : cfg["in_fft_conv1d_kernel_size"],
            "d_model"       : cfg["in_fft_output_size"],
            "p_dropout"     : cfg["p_in_fft_dropout"],
            "p_dropatt"     : cfg["p_in_fft_dropatt"],
            "p_dropemb"     : cfg["p_in_fft_dropemb"],
        },

        # ── Decoder (out_fft) ────────────────────────────────────
        "output_fft": {
            "n_layers"      : cfg["out_fft_n_layers"],
            "n_head"        : cfg["out_fft_n_heads"],
            "d_head"        : cfg["out_fft_d_head"],
            "d_inner"       : cfg["out_fft_conv1d_filter_size"],
            "kernel_size"   : cfg["out_fft_conv1d_kernel_size"],
            "d_model"       : cfg["out_fft_output_size"],
            "p_dropout"     : cfg["p_out_fft_dropout"],
            "p_dropatt"     : cfg["p_out_fft_dropatt"],
            "p_dropemb"     : cfg["p_out_fft_dropemb"],
        },

        # ── Duration predictor ───────────────────────────────────
        "duration_predictor": {
            "kernel_size"   : cfg["dur_predictor_kernel_size"],
            "filter_size"   : cfg["dur_predictor_filter_size"],
            "n_layers"      : cfg["dur_predictor_n_layers"],
            "p_dropout"     : cfg["p_dur_predictor_dropout"],
        },

        # ── Pitch predictor ──────────────────────────────────────
        "pitch_predictor": {
            "kernel_size"    : cfg["pitch_predictor_kernel_size"],
            "filter_size"    : cfg["pitch_predictor_filter_size"],
            "n_layers"       : cfg["pitch_predictor_n_layers"],
            "p_dropout"      : cfg["p_pitch_predictor_dropout"],
            "emb_kernel_size": cfg["pitch_embedding_kernel_size"],
        },

        # ── Energy predictor ─────────────────────────────────────
        "energy_predictor": {
            "kernel_size"    : cfg["energy_predictor_kernel_size"],
            "filter_size"    : cfg["energy_predictor_filter_size"],
            "n_layers"       : cfg["energy_predictor_n_layers"],
            "p_dropout"      : cfg["p_energy_predictor_dropout"],
            "emb_kernel_size": cfg["energy_embedding_kernel_size"],
            "conditioning"   : cfg["energy_conditioning"],
        },

        # ── Unused at inference ──────────────────────────────────
        "train_ds"      : None,
        "validation_ds" : None,
        "optim"         : None,
    })
    print(f"      Config sections: {[k for k in nemo_cfg.keys() if not k.endswith('_ds') and k != 'optim']}")
    return nemo_cfg


# ── Strip DataParallel prefix and return clean state_dict ─────────────────────
def clean_state_dict(raw_state: dict) -> dict:
    cleaned = {k.replace("module.", "", 1): v for k, v in raw_state.items()}
    stripped = sum(1 for k in raw_state if k.startswith("module."))
    if stripped:
        print(f"      Stripped 'module.' prefix from {stripped} keys.")
    return cleaned


# ── Package config + weights into .nemo (tar.gz) ─────────────────────────────
def save_nemo(nemo_cfg, state_dict: dict, nemo_path: str):
    print(f"[4/5] Packaging .nemo → {nemo_path}")
    tmp = Path(tempfile.mkdtemp())
    try:
        # Write config YAML
        cfg_file = tmp / "model_config.yaml"
        OmegaConf.save(nemo_cfg, cfg_file)
        print(f"      Config YAML: {cfg_file.stat().st_size / 1e3:.1f} KB")

        # Write weights
        weights_file = tmp / "model_weights.ckpt"
        torch.save(state_dict, weights_file)
        print(f"      Weights    : {weights_file.stat().st_size / 1e6:.1f} MB")

        # Bundle into .nemo
        with tarfile.open(nemo_path, "w:gz") as tar:
            tar.add(cfg_file,     arcname="model_config.yaml")
            tar.add(weights_file, arcname="model_weights.ckpt")

    finally:
        shutil.rmtree(tmp)

    final_size = os.path.getsize(nemo_path) / 1e6
    print(f"      .nemo size : {final_size:.1f} MB")


# ── Verify the .nemo can be read back and config is intact ────────────────────
def verify_nemo(nemo_path: str):
    print("[5/5] Verifying .nemo...")
    tmp = Path(tempfile.mkdtemp())
    try:
        with tarfile.open(nemo_path, "r:gz") as tar:
            members = tar.getnames()
            tar.extractall(tmp)

        assert "model_config.yaml" in members, "Missing model_config.yaml in archive"
        assert "model_weights.ckpt" in members, "Missing model_weights.ckpt in archive"

        cfg_check = OmegaConf.load(tmp / "model_config.yaml")
        for key in ["n_symbols", "n_speakers", "n_mel_channels", "padding_idx"]:
            assert hasattr(cfg_check, key), f"Config missing key: {key}"

        state_check = torch.load(tmp / "model_weights.ckpt", map_location="cpu")
        n_params = sum(v.numel() for v in state_check.values())
        print(f"      Archive members : {members}")
        print(f"      Config keys OK  : n_symbols={cfg_check.n_symbols}, "
              f"n_speakers={cfg_check.n_speakers}")
        print(f"      Weight params   : {n_params:,}")
    finally:
        shutil.rmtree(tmp)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    pt_path   = os.path.abspath(args.input)
    nemo_path = args.output or str(Path(pt_path).with_suffix(".nemo"))

    if not os.path.exists(pt_path):
        print(f"ERROR: Input file not found: {pt_path}", file=sys.stderr)
        sys.exit(1)

    if not pt_path.endswith(".pt"):
        print(f"WARNING: Input file does not have .pt extension: {pt_path}")

    print("=" * 60)
    print("  DLE FastPitch  .pt  →  .nemo  converter")
    print("=" * 60)
    print(f"  Input  : {pt_path}")
    print(f"  Output : {nemo_path}")
    print("=" * 60)

    ckpt, cfg       = load_pt(pt_path)
    validate_config(cfg)
    nemo_cfg        = build_nemo_cfg(cfg)
    state_dict      = clean_state_dict(ckpt["state_dict"])
    save_nemo(nemo_cfg, state_dict, nemo_path)
    verify_nemo(nemo_path)

    print("=" * 60)
    print(f"  ✅  Done!  →  {nemo_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()