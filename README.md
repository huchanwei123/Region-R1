# Region-R1

Train a Vision-Language Model (Qwen2.5-VL-3B) to intelligently crop query images for improved image retrieval, using our designed r-GRPO.

Given a query image and a user question, the model decides whether to crop a specific region to better match relevant candidates in a retrieval database. It outputs either a bounding box via `image_zoom_in_tool` or `NO_CROP_NEEDED`. Training uses EVA-CLIP-8B as the reward model, measuring whether cropping improves retrieval metrics (MRR, NDCG, rank, margin).

## Requirements

- Python 3.10+
- CUDA-capable GPU (tested on A100/H100)

### Dependencies

```bash
pip install torch torchvision
pip install transformers peft trl
pip install qwen-vl-utils
pip install datasets
pip install wandb
pip install Pillow matplotlib pandas numpy
pip install faiss-gpu  # or faiss-cpu
```

## Quick Start

### 1. Prepare Data

Create a re-ranking dataset from InfoSeek or E-VQA knowledge base with EVA-CLIP hard negatives:

```bash
# Set dataset name: "InfoSeek" (default) or "EVQA"
export DATASET_NAME=InfoSeek

# Set external data path (optional, defaults to ~/Desktop/<DATASET_NAME>-data)
export DATA_ROOT=/path/to/dataset-data

python prepare_data.py --num_entries 1000 --visualize_samples 20
```

| Argument | Default | Description |
|---|---|---|
| `--num_entries` | 1000 | Number of dataset entries to create |
| `--similarity_tolerance` | 0.05 | Hard negative similarity tolerance |
| `--visualize_samples` | 20 | Number of sample visualizations to generate |
| `--device` | `cuda:0` | CUDA device |

**Output:** `Reranker_Dataset/<DATASET_NAME>_top10_parquet/` (train.parquet, val.parquet)

### 2. Train

```bash
python main.py --reward_type mixture
```

| Argument | Default | Description |
|---|---|---|
| `--reward_type` | `mixture` | Reward function: `mixture` (recommended) or `absolute` |
| `--resume_from_checkpoint` | None | Path to checkpoint directory to resume from |
| `--log_level` | `INFO` | Logging level: DEBUG, INFO, WARNING, ERROR |
| `--weight_mrr` | 0.2 | Weight for delta-MRR in mixture reward |
| `--weight_ndcg` | 0.2 | Weight for delta-NDCG |
| `--weight_rank` | 0.2 | Weight for delta-rank |
| `--weight_margin` | 0.4 | Weight for delta-margin |
| `--initial_encouragement` | 0.0 | Initial encouragement bonus for cropping |
| `--encouragement_decay_steps` | 5000 | Steps for encouragement to decay to zero |

Training logs to both TensorBoard and Weights & Biases. Checkpoints are saved to `runs/<RUN_NAME>/checkpoints/`.

### 3. Evaluate

Evaluate a single checkpoint:

```bash
python evaluate_cropping.py --checkpoint runs/<run_name>/checkpoints/checkpoint-XXX
```

Or use the convenience wrapper:

```bash
python run_eval.py --checkpoint runs/<run_name>/checkpoints/checkpoint-XXX --num_samples 100
```

| Argument | Default | Description |
|---|---|---|
| `--checkpoint` | Final model | Path to checkpoint to evaluate |
| `--num_samples` | All | Number of validation samples |
| `--split` | `val` | Dataset split (`val` or `test`) |
| `--output` | Auto | Output CSV path |
| `--viz_dir` | Auto | Directory for improvement visualizations |
| `--wandb` | Off | Log results to W&B |

Example with shell script:

```bash
export CUDA_VISIBLE_DEVICES=7
bash run_eval.sh
```

## Project Structure

```
Region-R1/
├── main.py                     # Training entry point
├── config.py                   # Configuration (paths, hyperparameters, prompts)
├── model.py                    # Model loading (Qwen2.5-VL-3B + LoRA, EVA-CLIP-8B)
├── data.py                     # Lazy-loading dataset from parquet
├── train.py                    # GRPO training loop
├── rewards.py                  # Reward functions (absolute, mixture)
├── utils.py                    # CLIP scoring, bbox parsing, MRR/NDCG metrics
├── inference.py                # Post-training inference/testing
├── logging_config.py           # Logging setup
├── prepare_data.py             # Dataset preparation (InfoSeek or E-VQA)
├── evaluate_cropping.py        # Checkpoint evaluation (MRR improvement)
├── evaluation_callback.py      # Training callback for eval during training
├── training_logger_callback.py # Training callback for metrics logging
├── run_eval.py                 # Convenience evaluation wrapper
└── run_eval.sh                 # Shell script for evaluation
```

## Configuration

All configuration is centralized in `config.py`:

| Setting | Description |
|---|---|
| `RUN_NAME` | Experiment name (determines output directory) |
| `MODEL_ID` | Base VLM model (`Qwen/Qwen2.5-VL-3B-Instruct`) |
| `EVACLIP_MODEL_ID` | Reward model (`BAAI/EVA-CLIP-8B`) |
| `TRAINING_CONFIG` | Learning rate, epochs, batch size, etc. |
| `LORA_CONFIG` | LoRA rank, alpha, dropout, target modules |
| `WANDB_CONFIG` | W&B project, entity, tags |
| `SYSTEM_PROMPT` | Prompt template for the cropping assistant |

## Reward Functions

**Absolute** (`--reward_type absolute`): Direct MRR/NDCG score of the cropped image against candidates.

**Mixture** (`--reward_type mixture`): Weighted combination of four delta metrics (crop vs. baseline):
- Delta-MRR: change in Mean Reciprocal Rank
- Delta-NDCG: change in Normalized Discounted Cumulative Gain
- Delta-Rank: improvement in rank of the correct candidate (log scale)
- Delta-Margin: change in similarity gap between positive and hardest negative

Optionally includes a linearly-decaying encouragement bonus to incentivize cropping early in training.
