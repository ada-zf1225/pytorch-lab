# pytorch-lab

PyTorch from scratch, built for ML/algorithm engineer interview prep.

## Setup

- macOS (MPS): `pip install -r requirements.txt`
- CUDA: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 && pip install -r requirements.txt`
- HX2 (Slurm): `module load anaconda3 && conda create -n torchlab python=3.12 -y && conda activate torchlab && pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 && pip install -r requirements.txt`

Verify: `python 00_env/env_check.py`

## Layout

- `00_env/` — install sanity check: versions, devices, matmul throughput, autograd
- `01_tensor/` — tensor basics: creation, indexing, broadcasting, views vs copies
- `02_autograd/` — autograd mechanics, custom backward, gradient checking
- `03_train_loop/` — the training loop written by hand: data, loss, optimizer, eval
- `04_modules/` — nn.Module internals, common layers implemented from scratch
- `05_attention_gpt/` — attention and a minimal GPT
- `06_scale/` — mixed precision, torch.compile, multi-GPU basics
- `interview/` — interview questions and worked answers
