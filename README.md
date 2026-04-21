# GRPO Fine-Tuning on IDEA-E Ethical Reasoning

This repository contains notebooks from a hackathon exploring reinforcement learning fine-tuning of Gemma 2 2B-IT on the **IDEA-E Ethical Reasoning** dataset using **Group Relative Policy Optimization (GRPO)** via [Google Tunix](https://github.com/google/tunix).

## Overview

[GRPO](https://arxiv.org/abs/2402.03300) is a memory-efficient RL algorithm for improving LLM reasoning. It eliminates the separate value-function model used in PPO by computing relative advantages within a group of sampled responses. We apply it here to improve ethical reasoning capabilities in Gemma 2 2B-IT.

Training was run on TPU v6e-1 via Google Colab.

## Repository Structure

```
ablation_notebooks_final/   # GRPO training runs (v1–v4 ablations)
evals_final/                # Evaluation notebooks for each training variant
```

| Version | Training | Evaluation |
|---------|----------|------------|
| v1 | `ablation_notebooks_final/grpo-ideae-gemma2-2b-v1.ipynb` | `evals_final/grpo-ideae-gemma2-2b-all_eval-v1.ipynb` |
| v2 | `ablation_notebooks_final/grpo-ideae-gemma2-2b-v2.ipynb` | `evals_final/grpo-ideae-gemma2-2b-all_eval-v2.ipynb` |
| v3 | `ablation_notebooks_final/grpo-ideae-gemma2-2b-v3.ipynb` | `evals_final/grpo-ideae-gemma2-2b-all_eval-v3.ipynb` |
| v4 | `ablation_notebooks_final/grpo-ideae-gemma2-2b-v4.ipynb` | `evals_final/grpo-ideae-gemma2-2b-all_eval-v4.ipynb` |
| v4 (extended) | — | `evals_final/grpo-ideae-gemma2-2b-all_eval-v4pt2.ipynb` |
| baseline | — | `evals_final/grpo-ideae-gemma2-2b-all_eval_base.ipynb` |

## Dependencies

- [`google-tunix`](https://pypi.org/project/google-tunix/) >= 0.1.3
- JAX / Flax
- Hugging Face `datasets`
- Google Colab with TPU runtime (v6e-1 recommended)

## Usage

Open any notebook in Google Colab with a TPU runtime. The notebooks are self-contained and install their own dependencies.

## License

This work is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). See [LICENSE](LICENSE) for details.

## Author

Kevin Power — MIT
