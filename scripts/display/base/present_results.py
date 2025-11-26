import fire
from pathlib import Path
from typing import Dict, List, Any
from tabulate import tabulate

# Add parent directory to path to import common module
import sys
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
    "arc_challenge": ["english", "mc"],
    "gsm8k": ["math", "gen"],
    "gsm8k_cot": ["math", "gen"],
    "gsm8k_reasoning_base": ["math", "gen"],
    "minerva_math": ["math", "gen"],
    "minerva_math_reasoning_base": ["math", "gen"],
    # "minerva_math500": ["math", "gen"],
    "hellaswag": ["english", "mc"],
    "mmlu": ["english", "mc"],
    "mmlu_generative": ["english", "gen"],
    "mmlu_arabic": ["arabic", "mc"],
    "truthfulqa_mc2": ["english", "mc"],
    "winogrande": ["english", "mc"],
    # "leaderboard_gpqa_diamond": ["english", "mc"],
    "gpqa_diamond_cot_zeroshot": ["english", "gen"],
    "gpqa_diamond_reasoning_base": ["english", "gen"],
    "bbh": ["english", "gen"],
    "mmlu_pro": ["english", "gen"],
    "mbpp": ["code", "gen"],
    # "mbpp_instruct": ["code", "gen"],
    "humaneval": ["code", "gen"],
    "humaneval_64": ["code", "gen"],
    "ifeval": ["english"],
    "piqa": ["english", "mc"]
}

# Baseline models for comparison
BASELINE_MODELS = {
    # f"{BASE_CHECKPOINT_DIR}/k2-65b": "k2-65b",
    # f"{BASE_CHECKPOINT_DIR}/llama3-70b": "llama3-70b",
    # f"{BASE_CHECKPOINT_DIR}/qwen2.5-32b": "qwen2.5-32b",
    # f"{BASE_CHECKPOINT_DIR}/qwen2.5-72b": "qwen2.5-72b",
    # f"{BASE_CHECKPOINT_DIR}/falcon-h1-34b": "falcon-h1-34b",
    # f"{BASE_CHECKPOINT_DIR}/llama3.1-70b": "llama3.1-70b",
    f"{BASE_CHECKPOINT_DIR}/deepseek-v3.1-base": "deepseek-v3.1-base",
    # f"{WORKSPACE_CHECKPOINT_DIR}/k2plus_stage4_attn512k_jais250k_tp8_bestfit_400nodes_new/checkpoints/checkpoint_0005000": "midtrain-stage4",
    # f"{WORKSPACE_CHECKPOINT_DIR}/k2plus_stage2.5_attn32k_jais250k_tp8/checkpoints/checkpoint_0010000": "midtrain-stage4.5",
    # f"{WORKSPACE_CHECKPOINT_DIR}/k2plus_stage3_attn128k_jais250k_tp8_bestfit/checkpoints/checkpoint_0017500": "midtrain-stage3",
    # f"{WORKSPACE_CHECKPOINT_DIR}/k2plus_stage2_attn64k_jais250k_tp8_bestfit_fix/checkpoints/checkpoint_0045000": "midtrain-stage2",
    # f"{WORKSPACE_CHECKPOINT_DIR}/k2plus_stage1_attn8k_jais250k_tp8/checkpoints/checkpoint_0135000": "midtrain-stage1",
    # f"{BASE_CHECKPOINT_DIR}/qwen3-14b-base": "qwen3-14b-base",
    # f"{WORKSPACE_CHECKPOINT_DIR}/k2plus_stage3_attn128k_jais250k_rope10m_tp8_bestfit/checkpoints/checkpoint_0017500": "stage3-rope10m",
    # f"{WORKSPACE_CHECKPOINT_DIR}/k2plus_stage4_attn512k_jais250k_rope10m_tp8_bestfit/checkpoints/checkpoint_0007500": "stage4-rope10m",
    # f"{BASE_CHECKPOINT_DIR}/vocab_trimmed/iter_1249000": "k2+-base",
}

# Model name aliases for easier reference
MODEL_NAME_ALIASES = {
    # "stage1_v1": "k2plus_data.v1_attn8k_jais250k_tp8",
    # "stage1_v2": "k2plus_data.v2_attn8k_jais250k_tp8",
    # "stage1_v3": "k2plus_data.v3_attn8k_jais250k_tp8",
    # "stage1_v4": "k2plus_data.v4_attn8k_jais250k_tp8",
    "stage1": "k2plus_stage1_attn8k_jais250k_tp8",
    # "stage2_v1": "k2plus_stage2_attn64k_jais250k_tp8_normal",
    # "stage2_v2": "k2plus_stage2_attn64k_jais250k_tp8_bestfit",
    "stage2": "k2plus_stage2_attn64k_jais250k_tp8_bestfit_fix",
    "stage3_rope10m": "k2plus_stage3_attn128k_jais250k_rope10m_tp8_bestfit",
    # "stage3": "k2plus_stage3_attn128k_jais250k_tp8_bestfit",
    # "stage4": "k2plus_stage4_attn512k_jais250k_tp8_bestfit_400nodes_new",
    # "stage4.5": "k2plus_stage2.5_attn32k_jais250k_tp8",
    "stage5": "k2plus_stage5_attn32k_jais250k_rope10m_tp8",
    "stage5_merged": "k2plus_stage5_attn32k_jais250k_rope10m_tp8_with_think_merged",
    "stage5_separated": "k2plus_stage5_attn32k_jais250k_rope10m_tp8_with_think_separated",
    "stage4_rope10m": "k2plus_stage4_attn512k_jais250k_rope10m_tp8_bestfit",
    "final": "k2plus_stage4_cont_attn512k_jais250k_rope10m_tp8_bestfit"
}

# Constants for result processing
CHECKPOINT_AVERAGE_COUNT = 4

# Category order for averages
CATEGORY_ORDER = ['gen', 'mc', 'english', 'math', 'code']

# Result extraction keys for different metrics (use common, extend if needed)
RESULT_EXTRACTION_KEYS = BASE_RESULT_EXTRACTION_KEYS.copy()

# Metric name aliases for better display (use common, extend if needed)
METRIC_DISPLAY_ALIASES = BASE_METRIC_DISPLAY_ALIASES.copy()

# Helper Functions
# ================

def get_model_name_base(model_path: str) -> str:
    """Extract model name from path for base models.
    
    Args:
        model_path: Path to the model directory
        
    Returns:
        Model name extracted from path
    """
    return model_path.split("/")[-1]


def resolve_model_name(alias: str) -> str:
    """Resolve model name alias to full model name.

    Args:
        alias: Model name alias

    Returns:
        Full model name
    """
    return MODEL_NAME_ALIASES.get(alias, alias)



def display_results(
    baseline_original_rows: List[List[Any]], 
    baseline_averages_rows: List[List[Any]], 
    k2_plus_original_rows: List[List[Any]], 
    k2_plus_averages_rows: List[List[Any]]
) -> None:
    """Display the results in two formatted tables.

    Args:
        baseline_original_rows: Original rows for baseline models
        baseline_averages_rows: Averages rows for baseline models
        k2_plus_original_rows: Original rows for k2+ models
        k2_plus_averages_rows: Averages rows for k2+ models
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

    # Sort k2+ rows by checkpoint number in descending order
    sorted_k2_plus_original_rows, sorted_k2_plus_averages_rows = sort_rows_by_checkpoint(
        k2_plus_original_rows, k2_plus_averages_rows
    )

    # Combine all rows for each table
    all_original_rows = public_baseline_original_rows + baseline_original_rows[-1:] + sorted_k2_plus_original_rows
    all_averages_rows = public_baseline_averages_rows + baseline_averages_rows[-1:] + sorted_k2_plus_averages_rows

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
        maxcolwidths=20
    ))
    
    print("\n=== AVERAGES TABLE ===")
    print(tabulate(
        all_averages_rows,
        headers=averages_headers,
        tablefmt="tsv",
        numalign="right",
        floatfmt=".2f",
        maxcolwidths=20
    ))

def main(model_name: str) -> None:
    """Main function to process and display model evaluation results.

    Args:
        model_name: Name or alias of the model to analyze
    """
    # Resolve model name alias
    full_model_name = resolve_model_name(model_name)

    # Get checkpoint directories
    checkpoint_dirs = get_checkpoint_directories(full_model_name, WORKSPACE_CHECKPOINT_DIR)

    # Process baseline models
    baseline_original_rows, baseline_averages_rows = process_model_results(
        list(BASELINE_MODELS.keys()),
        METRICS_CONFIG,
        RESULT_EXTRACTION_KEYS,
        CATEGORY_ORDER,
        BASELINE_MODELS,
        CHECKPOINT_AVERAGE_COUNT,
        get_model_name_base,
        replace_checkpoint_name=True
    )

    # Process k2+ model checkpoints
    k2_plus_original_rows, k2_plus_averages_rows = process_model_results(
        checkpoint_dirs,
        METRICS_CONFIG,
        RESULT_EXTRACTION_KEYS,
        CATEGORY_ORDER,
        BASELINE_MODELS,
        CHECKPOINT_AVERAGE_COUNT,
        get_model_name_base,
        replace_checkpoint_name=True
    )

    # Display results
    display_results(baseline_original_rows, baseline_averages_rows, k2_plus_original_rows, k2_plus_averages_rows)


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
