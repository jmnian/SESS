# SESS: Submodular Evaluation Subset Selection in Automatic Prompt Optimization

This repository contains the code for the paper:

> **[Submodular Evaluation Subset Selection in Automatic Prompt Optimization](https://arxiv.org/abs/2601.03493)**  
> Jinming Nian, Zhiyuan Peng, Hongwei Shang, Dae Hoon Park, Yi Fang  
> *arXiv:2601.03493*

## Installation

### Requirements

- Python >= 3.10
- CUDA-compatible GPU (for vLLM inference)

### Setup

```bash
# Clone the repository
git clone https://github.com/jmnian/SESS.git
cd SESS

# Install dependencies using uv (recommended)
pip install uv
uv sync

# Or using pip
pip install -e .
```

### Download Datasets

The datasets are not included in the repository. Use the provided scripts to download them:

```bash
# Set your HuggingFace token (required for GPQA - it's a gated dataset)
export HF_TOKEN="your_huggingface_token"

# Download GPQA dataset
python scripts/download_gpqa_dataset.py

# Download MATH dataset
python scripts/download_math_dataset.py
```

GSM8K is downloaded automatically when first used.

## Usage

### Running Prompt Optimization with SESS

The main entry point is `run_opro_parallel.py`:

```bash
python run_opro_parallel.py \
    --dataset="gsm8k" \
    --subset_select_method="confidence_weighted_representative" \
    --subset_portion=3.5 \
    --num_search_steps=100 \
    --scorer_model="Qwen/Qwen2.5-7B-Instruct"
```

### Subset Selection Methods

| Method | `--subset_select_method` | Description |
|--------|--------------------------|-------------|
| Random | `random` | Uniformly random sampling (baseline) |
| Representative | `representative` | Facility location for diversity |
| Least Confident | `least_confident` | Select samples with lowest model confidence |
| Verbal Least Confident | `verbal_least_confident` | Select based on verbalized confidence |
| Confidence-Weighted Representative | `confidence_weighted_representative` | Submodular selection weighted by confidence |

### Example Experiments

```bash
# GSM8K with confidence-weighted representative selection (3.5% of training data)
python run_opro_parallel.py \
    --dataset="gsm8k" \
    --subset_select_method="confidence_weighted_representative" \
    --subset_portion=3.5 \
    --confidence_weight=0.5 \
    --alpha=0.7 \
    --num_search_steps=100

# MATH with representative selection
python run_opro_parallel.py \
    --dataset="math" \
    --subset_select_method="representative" \
    --subset_portion=3.5 \
    --num_search_steps=100

# GPQA-Diamond with least confident selection
python run_opro_parallel.py \
    --dataset="gpqa" \
    --task="diamond" \
    --subset_select_method="least_confident" \
    --subset_portion=10 \
    --num_search_steps=100
```

Or use the experiment runner script:

```bash
bash run_experiment.bash
```

## Results

Experiment results are stored in `outputs/optimization-results/`. Each experiment directory contains:
- `configs_dict.json`: Experiment configuration
- `test_evaluation_results.json`: Test set evaluation results

## Citation

If you find this code useful, please cite our paper:

```bibtex
@article{nian2026submodular,
  title={Submodular Evaluation Subset Selection in Automatic Prompt Optimization},
  author={Nian, Jinming and Peng, Zhiyuan and Shang, Hongwei and Park, Dae Hoon and Fang, Yi},
  journal={arXiv preprint arXiv:2601.03493},
  year={2026}
}
```

## Acknowledgments

This codebase builds upon [OPRO (Large Language Models as Optimizers)](https://arxiv.org/abs/2309.03409) by Yang et al. We thank the authors for releasing their code.

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.
