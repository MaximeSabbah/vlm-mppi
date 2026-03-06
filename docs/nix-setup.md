# Nix Setup Guide

## Prerequisites

Install Nix with flakes enabled:

```bash
# Install Nix
curl --proto '=https' --tlsv1.2 -sSf -L https://install.determinate.systems/nix | sh

# Nix with flakes is enabled by default with the Determinate installer.
# If using the official installer, add to ~/.config/nix/nix.conf:
#   experimental-features = nix-command flakes
```

## Usage

```bash
# Enter the development shell (creates venv, installs deps)
nix develop

# Run examples
python examples/01_test_vlm_basic.py --image scene.jpg
```

## How it works

The `flake.nix` creates a shell with:
- Python 3.10
- System libraries needed by OpenCV, PyTorch, etc.
- A Python venv (`.venv/`) created on first entry
- All pip dependencies from `requirements.txt` installed into the venv

The venv approach means Nix handles system deps while pip handles Python packages.
This avoids the complexity of nixifying every Python ML package.

## CUDA Support

The flake sets `config.cudaSupport = true`. For this to work, you need:
- NVIDIA drivers installed on the host
- The `nixpkgs-unfree` channel or `allowUnfree = true`

If you have issues with CUDA, install PyTorch manually after entering the shell:

```bash
nix develop
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## Pinning nixpkgs

The `flake.lock` file pins exact versions. To update:

```bash
nix flake update
```
