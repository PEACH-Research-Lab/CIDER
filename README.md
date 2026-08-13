# CIDER: Contextual Disclosure Boundaries for Privacy Preference Alignment

[![Paper](https://img.shields.io/badge/Paper-CIDER.pdf-b31b1b.svg)](https://arxiv.org/abs/2608.09164)
[![Python](https://img.shields.io/badge/python-3.11+-3776AB?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Dataset-FFD21E)](https://huggingface.co/datasets/peach-lab/CIDER)

<p align="center">
  <a href="CIDER.pdf"><img src="assets/overview.png" alt="CIDER overview" width="90%"></a>
</p>

## 📚 Contents

- ℹ️ [About](#-about)
- 📁 [Repository Layout](#-repository-layout)
- 🚀 [Quick start](#-quick-start)
- 📊 [Dataset Exploration](#-dataset-exploration)
- 🔬 [Evaluate LMs](#-evaluate-lms)
- 🔍 [Analyze Predictions](#-analyze-predictions)
- 💡 [Considerations](#-considerations)
- ⚖️ [License](#-license)
- 📝 [Citation](#-citation)

## **ℹ️ About**

This repository contains the **CIDER** dataset and evaluation harness for the COLM 2026 paper ***CIDER: A Dataset of Contextual Disclosure Boundaries for Privacy Preference Alignment.*** 

## What can you do with CIDER?

1. **Explore individual privacy disclosure preferences.** Each scenario provides nine disclosure
variants on a granularity × identifiability grid (G1-G3 × I1-I3). Participants rate YES/NO willingness to disclose from a role perspective and AI-mediated condition. Each participant has at least 8 diverse scenarios to depict how their disclosure preference manifest contextually.
2. **Evaluate LMs with in-context learning prediction tasks.** Please see our codebase for the evaluation set-up.
3. **Reuse or extend new study material with the toolkit.** You are welcome to reuse the study toolkit for further data collection or extend the dataset by generating nine disclosure variants for new scenarios. Please see our codebase for the generation pipeline.

## **📁 Repository layout**

```python
CIDER/                          # repo root
├── README.md
├── requirements.txt
├── .env.example
├── assets/overview.png
│
├── dataset/
│   ├── README.md
│   ├── original/               # study data (scenarios, ratings, visual cards)
│   ├── extensibility/          # generate new variants
│   └── quick_start.py / .ipynb
│
├── evaluation/
│   ├── configs/sample.json
│   ├── quick_start.py / .ipynb
│   ├── helper/                 # shared eval plumbing
│   │   ├── config.py
│   │   ├── constants.py
│   │   ├── load_data.py
│   │   ├── metrics.py
│   │   ├── plotting.py
│   │   ├── providers.py
│   │   ├── runner.py           # ICL prediction runner
│   │   ├── utils.py
│   │   └── clients/            # openai, openrouter, _template
│   ├── model_eval/             # run_eval, prompt sensitivity
│   ├── analysis/               # boundary-level accuracy, FP/FN
│   └── prompts/                # construction, templates, preview, sensitivity
│
└── outputs/                    # eval dumps, gitignored (<model>/<batch-id>/k{N}_pr{R}/)
```

## **🚀 Quick start**

We provide two quick start entry points for (1) dataset exploration and (2) model evaluation, respectively.

From the repo root, use **Python 3.11+**, then:

```python
pip install -r requirements.txt
cp .env.example .env   *# add API keys when you need generation or model calls*
```

### **1. Dataset exploration**

You can browse CIDER dataset with simple stats and (optionally) generate new disclosure variants.

**Notebook (interactive):** open `dataset/quick_start.ipynb`

**Script:**

```python
# Explore tables + basic analysis (no API key)
python dataset/quick_start.py

# Also generate 9 variants (needs OPENAI_API_KEY or OPENROUTER_API_KEY in .env)
python dataset/quick_start.py --generate-a   # Choice A: natural-language description
python dataset/quick_start.py --generate-b   # Choice B: PrivacyLens seed + story
```

### **2. Model evaluation**

You can preview prompts, run ICL predictions, analyze results, and optionally test prompt sensitivity.

Before that, make sure you:

1. Set up the model evaluation configuration `.json`: Edit or create your own config file referring to `evaluation/configs/sample.json` .
2. Set the API key named by each model’s `api_key_env` in `.env`.

**Notebook (interactive):** open `evaluation/quick_start.ipynb` and run §1 → §2 → §4 (optional §3 prompt sensitivity; §5 for any existing prediction folder).

**Script (preview + confirm + predict):**

```python
python evaluation/quick_start.py
python evaluation/quick_start.py --yes              *# skip confirm*
python evaluation/quick_start.py --preview-only     *# prompts only*
python evaluation/quick_start.py --analyze          *# also run analysis*
```

Or call the pieces directly:

```python
python evaluation/model_eval/run_eval.py --config evaluation/configs/sample.json
python evaluation/model_eval/run_prompt_sensitivity.py --config evaluation/configs/sample.json
python evaluation/analysis/boundary_level_accuracy.py --config evaluation/configs/sample.json
```

## 📊 Dataset Exploration

**Exploratory Analysis**

You can use the dataset here or download it from Hugging Face. Please place it under dataset/original/ directory to run the exploratory analysis of dataset. 


**(Optional) Extensibility**

You can generate 9 variants for a given new scenario described in natural language, or for a new scenario sourced from PrivacyLens dataset.

We also provide a simple interface to create new visual card artifacts.

Please refer to the paper for detailed study design.

## **🔬** Evaluate LMs

**Preview Prompts**

You can preview the system/user prompts for your selected setups on one sample case:

```python
python evaluation/quick_start.py \
--config evaluation/configs/<config.json> \
--preview-only
```

This prints the prompts and saves a copy under `<output_root>/prompts/<batch_id>/k{N}_pr{R}/example_prompts.txt`.



**Model Evaluation**

You can use CIDER to evaluate a model’s capability of individual-level privacy preference understanding.

run in-context learning predictions for the models and setups with config.json. 

```python
python evaluation/model_eval/run_eval.py \
--config evaluation/configs/<config.json>
```

- `--config`: configure data paths, models, setups, `batch_id`, `output_root`, and dataset settings, reasoning.



**(Optional) Prompt Sensitivity Evaluation**

You can experiment with a two other different prompt versions to measure under HC condition, how stable HC predictions are across paragphrased prompts for different models.

```python
python evaluation/model_eval/run_prompt_sensitivity.py \
--config evaluation/configs/<config.json>
```

- `-config`: configure data paths, models, setups, `batch_id`, `output_root`, and dataset settings, reasoning.

For the variant, you need to use setup “HC” and select the prompt "prompt_variants": ["v0", "v1", "v2"]

## 🔍 Analyze Predictions

**Boundary-level Accuracy Analysis**

You can conduct boundary-level analysis of accuracies for predictions of current configurations or specified predictions. Result CSVs and accompanied line and bar charts are output and saved by default.

```python
python evaluation/analysis/boundary_level_accuracy.py \
--runs-root <result_root> \
--batch-id <batch_id> \
--ks 1,4,5,6 \
--pr 6
```

- `--runs-root`: Folder with `<model>/<batch-id>/k{N}_pr{R}/*.jsonl` (`pr` = prediction reference / `icl_reference_k`)
- `--batch-id`: Batch name (required with `--runs-root`)
- `--ks` / `--pr`: Required together with `--runs-root` — `--ks` is comma-separated `n_icl` value(s) to score (setup-trend plots one figure per k); `--pr` is the `icl_reference_k` case set to load (`k*_pr{R}`).
- `--no-filter-pr-cases`: Skip filtering prediction rows to the `icl_reference_k` case set (default: filter on).
- `--annotate-n` / `--no-annotate-n`: Annotate accuracy (%) on plot points/bars (default on).

Or use a config (fills in paths, models, setups, and k for you):

```python
python evaluation/analysis/boundary_level_accuracy.py \
--config evaluation/configs/<config_file_name.json>
```

Useful options:

- `--models` / `--setups`: Limit to specific models or setups (comma-separated)
- `--no-plots`: CSV only, no figures
- `--output-dir`: Where to save results (default: `<runs-root>/analysis`)



**Variant-level FP/FN Analysis**

You can inspect per-variant false positive (FP) / false negative (FN) rates for predictions of current configurations or specified predictions. Result CSVs and accompanied FP/FN shift scatter plots (default: Baseline → HC) are saved by default.

```python
python evaluation/analysis/variant_level_fp_fn.py \
--runs-root <result_root> \
--batch-id <batch_id> \
--ks 1,4,5,6 \
--pr 6
```

- `--runs-root`: Folder with `<model>/<batch-id>/k{N}_pr{R}/*.jsonl`
- `--batch-id`: Batch name (required with `--runs-root`)
- `--ks` / `--pr`: Required together with `--runs-root` — `--ks` is comma-separated `n_icl` value(s) to score (shift figures use the last one); `--pr` is the `icl_reference_k` case set to load (`k*_pr{R}`).
- `--no-filter-pr-cases`: Skip filtering prediction rows to the `icl_reference_k` case set (default: filter on).

Or use a config:

```python
python evaluation/analysis/variant_level_fp_fn.py \
--config evaluation/configs/<config_file_name.json>
```

Useful options:

- `--from-setup` / `--to-setup`: Shift setup pair (default: `Baseline` → `HC`)
- `--models` / `--setups`: Limit to specific models or setups
- `--no-plots`: Tables only, no scatter figures
- `--output-dir`: Where to save results (default: `<runs-root>/analysis`)

## 💡 Considerations

- Scenarios describe sensitive disclosure situations (health, legal, relationship, and safety topics) by design; all named individuals are fictional.
- Participant data is de-identified and limited to coarse demographics.

## ⚖️ License

This work is licensed under the [MIT License](LICENSE).


## 📝 Citation

Please cite our paper if you find the code or dataset useful.

```
@article{guo2026cider,
  title   = {CIDER: Contextual Disclosure Boundaries for Privacy Preference Alignment},
  author  = {Guo, Bingcan and Xu, Eryue and Zhou, Jijie and Zhang, Zhiping and Li, Tianshi},
  journal = {arXiv preprint arXiv:2608.09164},
  year    = {2026}
}
```
