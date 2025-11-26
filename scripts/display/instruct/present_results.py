import fire
import sys
from pathlib import Path
from typing import Dict, List, Any, Union
from tabulate import tabulate

# Add parent directory to path to import common module
sys.path.insert(0, str(Path(__file__).parent.parent))

from common import (
    BASE_CHECKPOINT_DIR,
    WORKSPACE_CHECKPOINT_DIR,
    RESULT_EXTRACTION_KEYS as BASE_RESULT_EXTRACTION_KEYS,
    METRIC_DISPLAY_ALIASES as BASE_METRIC_DISPLAY_ALIASES,
    process_model_results,
    get_checkpoint_directories,
    generate_original_table_headers,
    generate_averages_table_headers,
    sort_rows_by_checkpoint
)

# Configuration Constants
# =====================

# Metrics configuration mapping task names to their categories
METRICS_CONFIG = {
    # "arc_challenge": ["english", "mc"],
    "gsm8k_reasoning_instruct": ["math", "gen"],
    "minerva_math_reasoning_instruct": ["math", "gen"],
    # "hellaswag": ["english", "mc"],
    # "mmlu": ["english", "mc"],
    # "mmlu_arabic": ["arabic", "mc"],
    "truthfulqa_mc2": ["english", "mc"],
    "winogrande": ["english", "mc"],
    # "leaderboard_gpqa_diamond": ["english", "mc"],
    # "gpqa_diamond_cot_zeroshot": ["science", "gen"],
    "gpqa_diamond_reasoning_instruct": ["english", "gen"],
    # "bbh": ["english", "gen"],
    "mmlu_pro": ["english", "gen"],
    "mbpp_instruct": ["code", "gen"],
    "humaneval_instruct": ["code", "gen"],
    "humaneval_64_instruct": ["code", "gen"],
    "ifeval": ["english", "gen"],
    # "aime24": ["math", "gen"],
    # "aime25": ["math", "gen"],
    # "piqa": ["english", "mc"]
}

# Baseline models for comparison
BASELINE_MODELS = {
    # f"{BASE_CHECKPOINT_DIR}/k2-65b": "k2-65b",
    # f"{BASE_CHECKPOINT_DIR}/llama3-70b": "llama3-70b",
    # f"{BASE_CHECKPOINT_DIR}/qwen2.5-32b": "qwen2.5-32b",
    # f"{BASE_CHECKPOINT_DIR}/qwen2.5-72b": "qwen2.5-72b",
    # f"{BASE_CHECKPOINT_DIR}/falcon-h1-34b": "falcon-h1-34b",
    # f"{BASE_CHECKPOINT_DIR}/llama3.1-70b": "llama3.1-70b",
    f"{BASE_CHECKPOINT_DIR}/qwen2.5-72b-instruct": "qwen2.5-72b-instruct",
    f"{BASE_CHECKPOINT_DIR}/k2-think": "llm360/k2-think",
    # f"{WORKSPACE_CHECKPOINT_DIR}/k2plus_stage4_attn512k_jais250k_tp8_bestfit_400nodes_new/checkpoints/checkpoint_0005000": "midtrain-stage4",
    # f"{WORKSPACE_CHECKPOINT_DIR}/k2plus_stage3_attn128k_jais250k_tp8_bestfit/checkpoints/checkpoint_0017500": "midtrain-stage3",
    # f"{WORKSPACE_CHECKPOINT_DIR}/k2plus_stage2_attn64k_jais250k_tp8_bestfit_fix/checkpoints/checkpoint_0045000": "midtrain-stage2",
    # f"{WORKSPACE_CHECKPOINT_DIR}/k2plus_stage1_attn8k_jais250k_tp8/checkpoints/checkpoint_0135000": "midtrain-stage1",
}

# Model name aliases for easier reference
MODEL_NAME_ALIASES = {
    "general": ["sft/mid4_sft_instruct", "sft/mid3_sft", "sft/mid4_sft_instruct_cos_epoch"],
    "reasoning": [
        # "sft/mid4_sft_reasoning_am",
        # "sft/mid4_sft_reasoning_ot",
        "sft/mid4_sft_reasoning_am_cos_epoch",
        # "sft/mid4_sft_reasoning_ot_cos_epoch",
        "sft/mid4_sft_reasoning_oss_cos_epoch",
        "sft/mid4.5_sft_reasoning_am_cos_epoch"
    ]
}

# Constants for result processing
CHECKPOINT_AVERAGE_COUNT = 1

# Category order for averages (includes science)
CATEGORY_ORDER = ['gen', 'mc', 'english', 'math', 'code', 'science']

# Result extraction keys for different metrics (use common, extend if needed)
RESULT_EXTRACTION_KEYS = BASE_RESULT_EXTRACTION_KEYS.copy()

# Metric name aliases for better display (use common, extend if needed)
METRIC_DISPLAY_ALIASES = BASE_METRIC_DISPLAY_ALIASES.copy()

# Helper Functions
# ================

def get_model_name_instruct(model_path: str) -> str:
    """Extract model name from path for instruct models.
    
    Args:
        model_path: Path to the model directory
        
    Returns:
        Model name extracted from path (includes parent directory)
    """
    return model_path.split("/")[-3] + "/" + model_path.split("/")[-1]


def resolve_model_name(alias: str) -> Union[str, List[str]]:
    """Resolve model name alias to full model name(s).

    Args:
        alias: Model name alias

    Returns:
        Full model name or list of model names
    """
    result = MODEL_NAME_ALIASES.get(alias, alias)
    return result if isinstance(result, list) else result

def display_results(
    baseline_original_rows: List[List[Any]], 
    baseline_averages_rows: List[List[Any]], 
    test_original_rows: List[List[Any]], 
    test_averages_rows: List[List[Any]]
) -> None:
    """Display the results in two formatted tables.

    Args:
        baseline_original_rows: Original rows for baseline models
        baseline_averages_rows: Averages rows for baseline models
        test_original_rows: Original rows for test models
        test_averages_rows: Averages rows for test models
    """
    # Sort baseline rows (excluding the last one which is midtrain-stage3)
    public_baseline_original_rows = baseline_original_rows[:-1]
    public_baseline_averages_rows = baseline_averages_rows[:-1]
    
    # Sort by average score for original table (using the avg column from averages table)
    public_baseline_original_rows.sort(
        key=lambda x: float(baseline_averages_rows[baseline_original_rows.index(x)][1]) 
        if baseline_averages_rows[baseline_original_rows.index(x)][1] != 'x' else 0, 
        reverse=True
    )
    public_baseline_averages_rows.sort(key=lambda x: float(x[1]) if x[1] != 'x' else 0, reverse=True)

    # Sort test rows by checkpoint number in descending order
    sorted_test_original_rows, sorted_test_averages_rows = sort_rows_by_checkpoint(
        test_original_rows, test_averages_rows
    )

    # Combine all rows for each table
    all_original_rows = public_baseline_original_rows + baseline_original_rows[-1:] + sorted_test_original_rows
    all_averages_rows = public_baseline_averages_rows + baseline_averages_rows[-1:] + sorted_test_averages_rows

    # Generate headers
    original_headers = generate_original_table_headers(METRICS_CONFIG, METRIC_DISPLAY_ALIASES)
    averages_headers = generate_averages_table_headers(CATEGORY_ORDER)

    # Display original metrics table
    print("=== ORIGINAL METRICS TABLE ===")
    print(tabulate(
        all_original_rows,
        headers=original_headers,
        tablefmt="tsv",
        numalign="right",
        floatfmt=".2f",
        maxcolwidths=60
    ))
    
    print("\n=== AVERAGES TABLE ===")
    print(tabulate(
        all_averages_rows,
        headers=averages_headers,
        tablefmt="tsv",
        numalign="right",
        floatfmt=".2f",
        maxcolwidths=60
    ))

def main(model_name: str) -> None:
    """Main function to process and display model evaluation results.

    Args:
        model_name: Name or alias of the model to analyze
    """
    # Resolve model name alias
    full_model_names = resolve_model_name(model_name)
    
    if isinstance(full_model_names, str):
        print(f"Processing model: {full_model_names}")

    # Get checkpoint directories
    checkpoint_dirs = get_checkpoint_directories(full_model_names, WORKSPACE_CHECKPOINT_DIR)

    # Process baseline models
    baseline_original_rows, baseline_averages_rows = process_model_results(
        list(BASELINE_MODELS.keys()),
        METRICS_CONFIG,
        RESULT_EXTRACTION_KEYS,
        CATEGORY_ORDER,
        BASELINE_MODELS,
        CHECKPOINT_AVERAGE_COUNT,
        get_model_name_instruct,
        replace_checkpoint_name=False
    )

    # Process test model checkpoints
    test_original_rows, test_averages_rows = process_model_results(
        checkpoint_dirs,
        METRICS_CONFIG,
        RESULT_EXTRACTION_KEYS,
        CATEGORY_ORDER,
        BASELINE_MODELS,
        CHECKPOINT_AVERAGE_COUNT,
        get_model_name_instruct,
        replace_checkpoint_name=False
    )

    # Display results
    display_results(baseline_original_rows, baseline_averages_rows, test_original_rows, test_averages_rows)


# Entry Point
# ===========

if __name__ == '__main__':
    try:
        fire.Fire(main)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
    except Exception as e:
        print(f"Error: {e}")
        raise
