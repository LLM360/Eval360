# Eval360

Eval360 is a long-context language model evaluation workspace built around the LM Evaluation Harness. It provides opinionated scripts and automation for benchmarking large checkpoints on reasoning, math, and code suites while coordinating large-cluster workflows (SLURM, Ray, and vLLM). The repository glues together local checkpoints, Hugging Face models, and multi-node serving endpoints to streamline end-to-end evaluation runs.

## Key Features
- Runs curated "base" vs "instruct" evaluation tracks with preset `lm_eval` task lists and few-shot settings.
- Launches batch jobs against SLURM- or Ray-backed vLLM servers for single- and multi-node inference.
- Ships utilities to download, convert, and organize checkpoints (Hugging Face pulls, internal xLLM conversions, etc.).
- Provides shared helpers and notebooks for aggregating results from large evaluation sweeps.

## Repository Layout
```
Eval360/
├── README.md
├── logs/                      # SLURM/vLLM logs and archived wandb diagnostics
├── models/                    # Local Hugging Face checkpoints (e.g. olmo-3-32b-think-sft)
├── scripts/
│   ├── convert/               # Checkpoint format converters
│   ├── display/               # Result summarizers (python + notebooks)
│   ├── download/              # Model download utilities
│   ├── eval/                  # Base & instruct evaluation launchers
│   └── serving/               # vLLM + Ray Serve job scripts and clients
└── outputs/                   # Evaluation artifacts (JSON results, samples, charts)
```

> The workspace contains additional submodules (`lm-evaluation-harness/`, `LOOM-Scope/`) that provide core evaluation code and long-context benchmark suites. They are managed separately and can be ignored when editing this README.

## Prerequisites
- Access to a SLURM-based GPU cluster (scripts expect `sbatch`, multi-GPU nodes, and optional multi-node allocations).
- Python 3.10+ environment with CUDA-capable dependencies; a Miniconda/Conda install is assumed in job scripts.
- `lm_eval` (LM Evaluation Harness), `vllm`, `ray`, `fire`, and Hugging Face libraries installed.
- Hugging Face access token for gated models (`HF_TOKEN` in download scripts).
- Optional: WANDB account for tracking runs (set `WANDB_API_KEY`) and OpenAI-compatible client libraries.

## Environment Setup
1. **Clone with submodules**
   ```bash
   git clone --recursive <repo-url> Eval360
   cd Eval360
   ```
2. **Create environment** (example)
   ```bash
   conda create -n eval360 python=3.10
   conda activate eval360
   pip install -r lm-evaluation-harness/requirements.txt
   pip install vllm ray[serve] fire
   ```
3. **Environment variables**  
   Set these in your shell or SLURM scripts as needed:
   - `HF_ALLOW_CODE_EVAL=1` (enable code eval tasks)
   - `VLLM_WORKER_MULTIPROC_METHOD=spawn`
   - `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` (for long-context inference)
   - `WANDB_API_KEY=<token>` if logging to Weights & Biases
   - `HF_TOKEN=<token>` for gated Hugging Face downloads
   - Modify `PATH` to point at your conda install (examples already in scripts)

## Downloading Models
Use `scripts/download/` utilities to fetch checkpoints:

- **Single model (Python helper)**
  ```bash
  python scripts/download/download_qwen.py \
    --model_name allenai/Olmo-3-32B-Think-SFT \
    --output_dir /lustrefs/users/suqi.sun/projects/Eval360/models/olmo-3-32b-think-sft
  ```

- **Batch download (SLURM job)**
  ```bash
  sbatch scripts/download/download_ckpts.sh
  ```
  The script iterates over a predefined list of models and calls `download_qwen.py`.

- **DeepSeek V3 snapshot**
  ```bash
  python scripts/download/download_deepseek_v3.py
  ```

Downloaded checkpoints land in `models/` by default; adjust paths if storing elsewhere.

## Serving Models
The `scripts/serving/` directory contains launchers for local vLLM endpoints and Ray Serve deployments:

- **Single-node vLLM**
  ```bash
  sbatch scripts/serving/serve_vllm.sh
  ```
  Starts `vllm serve` on a node, exporting the specified model over HTTP.

- **Multi-node Ray cluster**
  ```bash
  sbatch scripts/serving/sbatch_ray.sh
  ```
  Boots a head node plus workers, then launches a vLLM Ray application. Customize environment variables for topology or NCCL settings.

- **Ray Serve app**
  ```python
  # scripts/serving/deepseek.py
  from ray import serve
  serve.run(llm_app)
  ```
  Configure `LLMConfig` for deployment-scale inference with autoscaling and custom engine kwargs.

- **API client**
  ```bash
  python scripts/serving/api_client.py --host localhost --port 8080
  ```
  Sanity-checks the OpenAI-compatible endpoint exposed by vLLM/Ray.

## Running Evaluations
All evaluation scripts ultimately shell out to `lm_eval` with preset tasks.

### Base Track (`scripts/eval/base/`)
- **Batch orchestrator**
  ```bash
  sbatch scripts/eval/base/eval_baselines.sh
  ```
  Iterates over metrics (`mmlu`, `arc_challenge`, `gsm8k`, etc.) for one or more checkpoints, writing results under `outputs/base_harness_all/<metric>_<shots>shots/`.

- **Task-specific launchers** such as `eval_mmlu.sh`, `eval_hellaswag.sh`, `eval_gsm8k_math.sh`. They:
  - Wait for checkpoints to finish preprocessing (`done.txt` sentinel).
  - Call `lm_eval --model vllm --model_args pretrained=<ckpt>,tensor_parallel_size=8,...`.
  - Log structured output and raw samples to each checkpoint’s `eval_results/` directory (mirrored under `outputs/` if using provided paths).
  - Require appropriate tensor parallel size, dtype, and generation kwargs (see script for defaults).

### Instruct Track (`scripts/eval/instruct/`)
- **All-in-one local launcher**
  ```bash
  sbatch scripts/eval/instruct/eval_harness_all_in-batch-vllm_local.sh
  ```
  Spins up a local vLLM server, waits for readiness (`curl` loop), then runs a configurable list of chat-based benchmarks (e.g., `ifeval`, `humaneval_instruct`) via `lm_eval --model local-chat-completions --apply_chat_template`.

- **Per-metric scripts** cover RULER, GPQA Diamond, AIME, GSM8K reasoning, MMLU Redux, etc. These mirror base scripts but set chat templates, system instructions, or reasoning-specific generation arguments.

- **Logging & outputs**
  - SLURM logs route to `logs/` with job-name tags.
  - `lm_eval` writes JSON metrics at `outputs/instruct_harness_all/<metric>_<shots>shots/` alongside sample generations.

### Common Options
- Adjust `tensor_parallel_size`, `gpu_memory_utilization`, and `max_gen_toks` according to your hardware.
- Pass `--confirm_run_unsafe_code` for tasks that execute model outputs (needed for code eval).
- Use WANDB integration (`--wandb_args`) if enabling in scripts.

## Converting Checkpoints
`scripts/convert/convert_k2_plus_midtrain_to_hf.sh` automates turning xLLM FSDP checkpoints into Hugging Face format.

Workflow highlights:
1. Loop over iterations (e.g., `checkpoint_0002500` to `checkpoint_0050000`).
2. Wait for source checkpoint directories to appear.
3. Run `tools/convert_checkpoint_format.py` and `tools/ckpt_convertion_xllm_to_hf.py` with the correct tokenizer, HF config, and rope scaling parameters.
4. Drop temporary `model.tp*.pt` shards after conversion.

Customize `TP`, tokenizer path, HF config, and checkpoint root before submission.

## Viewing Results
- `scripts/display/common.py` collects shared logic for parsing `eval_results` directories, deriving per-task metrics, and computing category averages.
- Jupyter notebooks under `scripts/display/` (e.g., `base/jsonl_viewer.ipynb`, `instruct/get_scores.ipynb`) load JSON results and logs to produce tables or trend plots.
- Generated artifacts:
  - `outputs/base_harness_all/.../results_*.json` – structured metrics per task.
  - `outputs/instruct_harness_all/.../samples_*.jsonl` – raw completions for qualitative inspection.

## Logs & Artifacts
- `logs/`: job outputs (`slurm_<jobname>_<id>.out`), Ray/vLLM diagnostics, archived logs for reruns.
- `outputs/`: canonical evaluation products. Directory names encode task, few-shot setting, and checkpoint path.
- `scripts/eval/**/wandb/`: cached wandb runs with metadata, requirements, and debug logs.

## Troubleshooting & Tips
- **Checkpoint readiness:** many scripts poll for `done.txt`; ensure preprocessing jobs create this sentinel or adjust the logic.
- **Long sequence lengths:** set `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` and ensure GPU memory utilization suits the model.
- **Concurrency tuning:** adjust `num_concurrent`, `batch_size`, and `max_gen_toks` in `lm_eval` arguments to avoid timeouts.
- **Ray cluster IPs:** `sbatch_ray.sh` auto-detects IPv6 vs IPv4; verify network interfaces if deployments hang.
- **Hugging Face permissions:** keep tokens in environment variables rather than hardcoding in scripts when possible.
- **Cleanup:** large conversions generate temporary shards—confirm scripts remove them or handle manual cleanup.

## Acknowledgements
Eval360 builds on the open-source LM Evaluation Harness and leverages vLLM, Ray Serve, and various Hugging Face model releases. Review the respective licenses and documentation for details on redistribution and usage.
