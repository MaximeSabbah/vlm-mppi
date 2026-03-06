# Licensing Analysis

This document tracks the license of every dependency to ensure
the full stack is safe for commercial use and IP filing.

## Summary

**All dependencies use permissive licenses (Apache 2.0, BSD, MIT).**
No copyleft (GPL) dependencies are included.

## VLM Models

| Model | License | Commercial Use | Notes |
|-------|---------|----------------|-------|
| Qwen2.5-VL-7B-Instruct | Apache 2.0 | ✅ Unrestricted | Recommended starting point |
| Qwen2.5-VL-32B-Instruct | Apache 2.0 | ✅ Unrestricted | Better accuracy |
| Qwen3-VL (all sizes) | Apache 2.0 | ✅ Unrestricted | Latest, best performance |
| Pixtral 12B (Mistral) | Apache 2.0 | ✅ Unrestricted | Alternative option |
| InternVL3 | MIT | ✅ Unrestricted | Alternative option |

### Models to AVOID for IP purposes

| Model | License | Issue |
|-------|---------|-------|
| Qwen2.5-VL-3B | Qwen Research License | ❌ No commercial use |
| Qwen2.5-VL-72B | Qwen License | ⚠️ <100M MAU limit |
| LLaMA 3.2 Vision | Meta Community License | ⚠️ <700M MAU, extra conditions |
| GPT-4V / Gemini / Claude | Proprietary API | ❌ No IP possible, API dependency |

## Python Dependencies

| Package | License | Type |
|---------|---------|------|
| PyTorch | BSD-3 | Permissive |
| transformers | Apache 2.0 | Permissive |
| accelerate | Apache 2.0 | Permissive |
| vLLM | Apache 2.0 | Permissive |
| OpenCV | Apache 2.0 | Permissive |
| NumPy | BSD-3 | Permissive |
| Pillow | MIT-like (HPND) | Permissive |

## Robotics Stack

| Package | License | Type |
|---------|---------|------|
| Pinocchio | BSD-2 | Permissive |
| Crocoddyl | BSD-3 | Permissive |
| mim_solvers | BSD-2 | Permissive |
| MuJoCo | Apache 2.0 | Permissive |

## What You CAN Patent

Your novel contributions (the architecture and method):
- The interface protocol between VLM outputs and MPPI cost functions
- How VLM scene understanding maps to collision constraints / goals
- Novel cost-shaping or reward-shaping for HRI
- Real-time integration pipeline (latency mismatch handling)
- Domain-specific fine-tuning methodology

## What You CANNOT Patent

- The VLM model itself (Alibaba's work)
- The MPPI algorithm (public domain, Williams et al.)
- The CaT mechanism (Chane-Sane et al.)
- Pinocchio / Crocoddyl algorithms

**Disclaimer:** This is not legal advice. Consult an IP attorney for patent filing.
