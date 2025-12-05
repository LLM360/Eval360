# Eval360

Eval360 is a long-context language model evaluation workspace built around the LM Evaluation Harness. It provides opinionated scripts and automation for benchmarking large checkpoints on reasoning, math, and code suites while coordinating large-cluster workflows (SLURM, Ray, and vLLM). The repository glues together local checkpoints, Hugging Face models, and multi-node serving endpoints to streamline end-to-end evaluation runs.

## Key Features
- Runs curated "base" vs "instruct" evaluation tracks with preset `lm_eval` task lists and few-shot settings.
- Launches batch jobs against SLURM- or Ray-backed vLLM servers for single- and multi-node inference.
- Ships utilities to download and organize checkpoints (Hugging Face pulls, local organization helpers).
- Provides shared helpers and notebooks for aggregating results from large evaluation sweeps.

## Repository Layout
```
Eval360/
├── README.md
├── scripts/
│   ├── display/               # Result summarizers (python + notebooks)
│   ├── download/              # Model download utilities
│   ├── eval/                  # Base & instruct evaluation launchers
│   └── serving/               # vLLM + Ray Serve job scripts and clients
```

> The workspace includes additional submodules (`lm-evaluation-harness/`, `LOOM-Scope/`) that supply core evaluation logic and long-context benchmark suites.

## Prerequisites
- Access to a SLURM-based GPU cluster (scripts expect `sbatch`, multi-GPU nodes, and optional multi-node allocations).
- Python 3.10+ environment with CUDA-capable dependencies; a Miniconda/Conda install is assumed in job scripts.
- `lm_eval` (LM Evaluation Harness), `vllm`, `ray`, `fire`, and Hugging Face libraries installed.
- Hugging Face access token for gated models (`HF_TOKEN` in download scripts).
- Optional: OpenAI-compatible client libraries if calling serving endpoints through the provided API client.

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
   - `HF_TOKEN=<token>` for gated Hugging Face downloads
   - Modify `PATH` to point at your conda install (examples already in scripts)

## Downloading Models
Use `scripts/download/` utilities to fetch checkpoints into a local directory of your choice:

- **Single model (Python helper)**
  ```bash
  python scripts/download/download_qwen.py
  ```
  Update the `MODEL_NAME`, `OUTPUT_DIR`, and `HF_TOKEN` constants at the top of the script before running it so the desired checkpoint and destination directory are used.

- **DeepSeek V3 snapshot**
  ```bash
  python scripts/download/download_deepseek_v3.py
  ```

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
  Iterates over metrics (`mmlu`, `arc_challenge`, `gsm8k`, etc.) for one or more checkpoints, writing results to the `OUTPUT_PATH` configured inside the script.

- **Metric-specific launchers**
  - `eval_mmlu.sh`, `eval_mmlu_pro.sh`, `eval_mmlu_arabic.sh` – sweep checkpoints across MMLU variants, waiting for `done.txt` before running and handling few-shot counts.
  - `eval_arc_bbh_gpqa_piqa.sh`, `eval_hellaswag.sh`, `eval_truthfulqa_winogrande_ifeval.sh` – group related benchmarks to share job resources while logging per-task outputs.
  - `eval_gsm8k_math.sh`, `eval_humaneval_mbpp.sh`, `eval_gpqa_diamond_gen.sh` – set custom generation parameters (max tokens, sampling) tuned for math and code evaluations.
  - `eval_harness_all_separate-vllm.sh` – launches a local vLLM server inside the job and iterates through a predefined metric list using the OpenAI-compatible completions API, running each metric in the background so multiple evaluations proceed concurrently.
  Each script can be launched independently with `sbatch`, taking the model name and optional iteration range as arguments; inspect the header comments to match expected positional parameters.

- **Shared behavior**  
  All per-metric scripts:
  - Wait for checkpoints to finish preprocessing (`done.txt` sentinel).
  - Call `lm_eval --model vllm --model_args pretrained=<ckpt>,tensor_parallel_size=8,...`.
  - Log structured output and raw samples to each checkpoint’s `eval_results/` directory and the configured `--output_path`.
  - Require appropriate tensor parallel size, dtype, and generation kwargs (see script for defaults).

### Instruct Track (`scripts/eval/instruct/`)
- **Preset sweep runner**
  - `eval_baselines.sh` mirrors the base variant but configures chat-oriented tasks and points to instruction-tuned checkpoints or served endpoints.
  - `eval_gsm8k_math.sh`, `eval_humaneval_mbpp.sh`, `eval_truthfulqa_winogrande_ifeval.sh`, `eval_ruler.sh`, `eval_gpqa_diamond_gen.sh`, `eval_aime.sh`, `eval_mmlu_redux.sh`, `eval_mmlu_pro.sh` – provide per-suite entry points with reasoning-aware prompts and sampling arguments tailored for instruction models.
  - `eval_harness_all.sh` stitches multiple scripts together for back-to-back execution on cluster nodes when batch-evaluating a single checkpoint.
  - `eval_harness_all_separate-vllm.sh` connects to an existing vLLM endpoint (or starts one if running locally) and streams through a curated set of instruction benchmarks, automatically switching between chat and completion APIs and batching tasks via background processes.
  Each script documents required environment variables (e.g., serving endpoints, port numbers) near the top; adjust these before submission.

- **Per-metric scripts** cover RULER, GPQA Diamond, AIME, GSM8K reasoning, MMLU Redux, etc. These mirror base scripts but set chat templates, system instructions, or reasoning-specific generation arguments.

- **Logging & outputs**
  - `lm_eval` writes metrics and samples to the `OUTPUT_PATH` configured inside each script.
  - Standard SLURM output/error files record runtime details; adjust the `#SBATCH --output` directives to match your logging location.

### Common Options
- Adjust `tensor_parallel_size`, `gpu_memory_utilization`, and `max_gen_toks` according to your hardware.
- Pass `--confirm_run_unsafe_code` for tasks that execute model outputs (needed for code eval).
- Enable optional tracking integrations by editing the scripts; none are required by default.

## Viewing Results
- `scripts/display/common.py` collects shared logic for parsing `eval_results` directories, deriving per-task metrics, and computing category averages.
- Utilities under `scripts/display/` can load structured JSON outputs and render tables or charts for quick inspection.
- Generated artifacts include JSON score summaries and optional sample dumps at the directories supplied via each script’s `--output_path`.

## Troubleshooting & Tips
- **Checkpoint readiness:** many scripts poll for `done.txt`; ensure preprocessing jobs create this sentinel or adjust the logic.
- **Long sequence lengths:** set `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` and ensure GPU memory utilization suits the model.
- **Concurrency tuning:** adjust `num_concurrent`, `batch_size`, and `max_gen_toks` in `lm_eval` arguments to avoid timeouts.
- **Ray cluster IPs:** `sbatch_ray.sh` auto-detects IPv6 vs IPv4; verify network interfaces if deployments hang.
- **Hugging Face permissions:** keep tokens in environment variables rather than hardcoding in scripts when possible.
- **Cleanup:** evaluation runs can emit large sample dumps and logs—prune older artifacts periodically to manage storage.

## Acknowledgements
Eval360 builds on the open-source LM Evaluation Harness and leverages vLLM, Ray Serve, and various Hugging Face model releases. Review the respective licenses and documentation for details on redistribution and usage.
