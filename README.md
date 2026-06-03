# SEA-Embedding: Open and Reproducible Text Embeddings for Southeast Asia

This repository contains the training code for **SEA-Embedding**, an open and reproducible text embedding model for Southeast Asian languages. The training pipeline is built on top of [Sentence Transformers](https://www.sbert.net) and consists of two stages: unsupervised pre-training (Step 1) and supervised fine-tuning (Step 2).

---

## Installation

Requires **Python 3.9+** and **PyTorch 1.11.0+**.

### Option 1: Conda

```bash
conda create -n sea-embedding python=3.10
conda activate sea-embedding

# Install the package in editable mode
pip install -e .

# Install additional training dependencies
pip install -r requirements.txt
```

### Option 2: uv

```bash
# Install uv if not already installed
pip install uv

# Sync all dependencies from the lockfile
uv sync

# Or install directly from requirements
uv pip install -r requirements.txt
uv pip install -e .
```

---

## Training

The training pipeline is split into two steps, each with a corresponding SLURM script for multi-node, multi-GPU execution.

### Step 1: Unsupervised Pre-training

Edit `slurm_run_multinode_step1.sh` to set your cluster parameters:

```bash
#SBATCH --nodes=NUMBER_OF_NODES      # number of nodes to use
#SBATCH --nodelist=YOUR_NODE         # specific node names
#SBATCH --gres=gpu:GPU               # number of GPUs per node
```

Then set the model and output name inside the script:

```bash
model="MODEL FROM HF"    # e.g. "aisingapore/sea-lion-7b"
model_name="model_name"  # used for output directory naming
```

Submit the job:

```bash
sbatch slurm_run_multinode_step1.sh
```

Logs are saved to `log/step1/step1-<job_id>.out`.

---

### Step 2: Supervised Fine-tuning

Edit `slurm_run_multinode_step2.sh` to set the model from Step 1:

```bash
model="MODEL FROM previous step"  # path or HF repo of Step 1 output
model_name="model_name"
```

Update the SLURM header as in Step 1, then submit:

```bash
sbatch slurm_run_multinode_step2.sh
```

Logs are saved to `log/step2/step2-<job_id>.out`.

---

### Key Hyperparameters

Both scripts sweep over the following hyperparameters by default:

| Parameter | Values |
|---|---|
| Learning rate | `1e-4`, `2e-4`, `1e-5`, `2e-5` |
| Temperature | `0.09`, `0.07`, `0.05`, `0.03`, `0.01` |
| Warmup proportion | `0.1` |
| Loss function | `MultipleNegativesSymmetricRankingLossReweighting` |
| LR scheduler | `CosineLR` |

---

## Citation

If you use SEA-Embedding in your work, please cite:

```bibtex
@misc{limkonchotiwat2026seaembeddingopenreproducibletext,
      title={SEA-Embedding: Open and Reproducible Text Embeddings for Southeast Asia}, 
      author={Peerat Limkonchotiwat and Raymond Ng and Sarana Nutanong and Jian Gang Ngui},
      year={2026},
      eprint={2606.03027},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2606.03027}, 
}
```
