# Physics Foundation Models – Local Testing Guide

This document is a quick practical guide to installing and testing a few physics / PDE foundation models locally.

## Models Covered

- [PDE-Transformer](#pde-transformer)
- [Walrus](#walrus)
- [GPhyT](#gphyt)
- [MixER / MoE-POT](#mixer--moe-pot)
- [PDEformer-2](#pdeformer-2)

---

## Quick Comparison

| Model            | Code & Pretrained Weights?                                          | Install & Usage (Summary)                                                                                     | Ease to Test Locally |
|------------------|---------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------|----------------------|
| PDE-Transformer  | Yes – pip package `pdetransformer`, docs, HF models collection.    | `pip install pdetransformer`, then `PDETransformer.from_pretrained(...)` on Hugging Face weights.             | ⭐⭐⭐ Easiest – most “production-ready” FM in the list; great starting point. |
| Walrus           | Yes – GitHub repo + HF model `polymathic-ai/walrus` and finetunes. | `git clone` repo + install; load weights from HF and run their demo notebooks.                                | ⭐⭐⭐ Easy – as long as you’re OK with ~10 GB on disk and ~16 GB VRAM. |
| GPhyT            | Yes – official GitHub repo + HF weights.                           | Clone repo, `pip install -e .`, download weights from HF, run provided examples.                              | ⭐⭐☆ Medium-easy – clean code but larger models; better with ≥16 GB VRAM. |
| MixER / MoE-POT  | Yes – NeurIPS repo + HF weights (tiny/small/medium).               | Clone repo, install requirements, load a tiny/small model from HF and run `evaluate.py`.                      | ⭐⭐☆ Medium-easy – more research-y scripts, but well documented. Start with tiny/small. |
| PDEformer-2      | Yes – official GitHub with pretrained 2D models.                   | Clone repo, follow README; use their code to query the model at arbitrary space-time points.                  | ⭐⭐☆ Medium-easy – more config and dataset handling, but straightforward if you follow examples. |

---

## PDE-Transformer

**Type:** PDE FM backbone  
**Ease:** ⭐⭐⭐ (Easiest – recommended starting point)

### Overview

PDE-Transformer is a production-ready PDE foundation model with:

- A dedicated **pip package** (`pdetransformer`)
- Multiple pretrained variants (small / base / large)
- Hugging Face model collection and examples

### Install

```bash
# Create / activate a virtual env if you like

pip install torch  # pick the right CPU/GPU wheel for your system
pip install pdetransformer
