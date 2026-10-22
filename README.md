# iris-3b-data-space-dmd

Few-step distillation of [Iris-3B](https://huggingface.co/speridlabs/iris-3b) using **data-space iteration** ([arXiv:2610.12102](https://arxiv.org/abs/2610.12102)): refine clean pixel predictions from fixed noise instead of a hand-tuned flow-time schedule, trained with a DMD/TDM-style teacher–student–fake setup (full fine-tune, not LoRA).

Work in progress. Training targets ~4-step generation at 1024px on a single H100 with memory optimizations (8-bit Adam, cached text embeddings, etc.). Detailed notes live in local docs (not necessarily in this repo).

**References:** [Iris-3B](https://huggingface.co/speridlabs/iris-3b) · [Paper](https://arxiv.org/abs/2610.12102) · [MOSS data-space DMD (community)](https://huggingface.co/lodestones/MOSS-Image-SFT-8-steps-dmd)
