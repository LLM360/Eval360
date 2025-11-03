import os
import fire
import glob
import json
import statistics
from typing import Dict, List, Tuple, Any
from tabulate import tabulate

# Configuration Constants
# =====================

# Base checkpoint directory
BASE_CHECKPOINT_DIR = "/lustrefs/users/runner/checkpoints/huggingface"
WORKSPACE_CHECKPOINT_DIR = "/lustrefs/users/runner/workspace/checkpoints/huggingface"

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

# Result extraction keys for different metrics
RESULT_EXTRACTION_KEYS = {
    'arc_challenge': 'acc_norm,none',
    'hellaswag': 'acc_norm,none',
    'leaderboard_gpqa_diamond': 'acc_norm,none',
    'gpqa_diamond_cot_zeroshot': 'exact_match,flexible-extract',
    'piqa': 'acc_norm,none',
    'mmlu': 'acc,none',
    'truthfulqa_mc2': 'acc,none',
    'winogrande': 'acc,none',
    'mmlu_arabic': 'acc,none',
    'mmlu_pro': 'exact_match,custom-extract',
    'bbh': 'exact_match,get-answer',
    'mbpp_instruct': 'pass_at_1,extract_code',
    'humaneval_instruct': 'pass@1,create_test',
    'humaneval_64_instruct': 'pass@64,create_test',
    'gsm8k_reasoning_instruct': 'math_verify,none',
    'minerva_math_reasoning_instruct': 'math_verify,none',
    'aime24': 'exact_match,none',
    'aime25': 'exact_match,none'
}

# Metric name aliases for better display
METRIC_DISPLAY_ALIASES = {
    'arc_challenge': 'ARC-C',
    'hellaswag': 'HellaSwag',
    'leaderboard_gpqa_diamond': 'GPQA-Diamond-MC',
    'gpqa_diamond_cot_zeroshot': 'GPQA-Diamond-CoT',
    'piqa': 'PIQA',
    'mmlu': 'MMLU',
    'truthfulqa_mc2': 'TruthfulQA',
    'winogrande': 'WinoGrande',
    'mmlu_arabic': 'MMLU-Arabic',
    'mmlu_pro': 'MMLU-Pro',
    'bbh': 'BBH',
    'mbpp_instruct': 'MBPP',
    'humaneval_instruct': 'HumanEval',
    'humaneval_64_instruct': 'HumanEval-64',
    'gsm8k_reasoning_instruct': 'GSM8K-Reasoning',
    'minerva_math_reasoning_instruct': 'Minerva-MATH-Reasoning',
    'ifeval': 'IFEval',
    'aime24': 'AIME24',
    'aime25': 'AIME25'
}

# Utility Functions
# =================

def calculate_average_format(value_list: List[float]) -> str:
    """Calculate and format the average of a list of values.

    Args:
        value_list: List of numeric values

    Returns:
        Formatted average as string with 2 decimal places
    """
    if 0 in value_list:
        return '0.00'
    return f'{statistics.mean(value_list):.2f}'


def calculate_mean(num_list: List[float]) -> float:
    """Calculate the mean of a list of numbers.

    Args:
        num_list: List of numeric values

    Returns:
        Mean value, or 0.00 if list is empty or contains 0
    """
    if not num_list or 0 in num_list:
        return 0.00
    return statistics.mean(num_list)


def extract_result_value(results: Dict[str, Any], metric: str) -> float:
    """Extract the appropriate result value for a given metric.

    Args:
        results: Dictionary containing metric results
        metric: Name of the metric to extract

    Returns:
        Extracted result value as float
    """
    if metric not in results:
        return 0.0

    # Handle special case for ifeval metric
    if metric == 'ifeval':
        return statistics.mean([
            results[metric]['prompt_level_strict_acc,none'],
            results[metric]['inst_level_strict_acc,none']
        ])

    # Use the predefined extraction key
    extraction_key = RESULT_EXTRACTION_KEYS.get(metric)
    if extraction_key and extraction_key in results[metric]:
        return results[metric][extraction_key]

    return 0.0


def load_model_results(model_path: str) -> Dict[str, Any]:
    """Load evaluation results for a single model.

    Args:
        model_path: Path to the model directory

    Returns:
        Dictionary containing all metric results for the model
    """
    results = {}

    # Find all result files for this model
    result_files = sorted(glob.glob(f'{model_path}/eval_results/*/*/results_*.json'))

    for result_file in result_files:
        try:
            with open(result_file, 'r') as f:
                data = json.load(f)

            for key, value in data['results'].items():
                if key in METRICS_CONFIG:
                    if key in results:
                        results[key] = {**results[key], **value}
                    else:
                        results[key] = value
        except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
            print(f"Warning: Could not load results from {result_file}: {e}")
            continue

    return results


def calculate_category_averages(outputs: Dict[str, float]) -> List[float]:
    """Calculate category averages for different types of tasks.

    Args:
        outputs: Dictionary mapping metric names to their scores

    Returns:
        List of averages in order: [gen_avg, mc_avg, english_avg, math_avg, code_avg]
    """
    category_sums = {
        'gen': [],
        'mc': [],
        'english': [],
        'math': [],
        'code': [],
        'science': []
    }

    for metric, score in outputs.items():
        categories = METRICS_CONFIG.get(metric, [])
        for category in categories:
            if category in category_sums:
                category_sums[category].append(score)

    # Return averages in the expected order
    return [
        calculate_mean(category_sums['gen']),
        calculate_mean(category_sums['mc']),
        calculate_mean(category_sums['english']),
        calculate_mean(category_sums['math']),
        calculate_mean(category_sums['code']),
        calculate_mean(category_sums['science'])
    ]


# Main Processing Functions
# =========================

def process_single_model(model_path: str, cache: List[float], count: int) -> Tuple[List, List, List[float], int]:
    """Process results for a single model.

    Args:
        model_path: Path to the model directory
        cache: List of cached values for averaging
        count: Current count for checkpoint averaging

    Returns:
        Tuple of (original_row_data, averages_row_data, updated_cache, updated_count)
    """
    # model_name = BASELINE_MODELS.get(model_path, model_path.split("/")[-1])
    model_name = BASELINE_MODELS.get(model_path, model_path.split("/")[-3] + "/" + model_path.split("/")[-1])

    # Load results for this model
    results = load_model_results(model_path)

    # Extract outputs for all metrics
    outputs = {
        metric: extract_result_value(results, metric) * 100
        for metric in METRICS_CONFIG
    }

    # Calculate category averages
    category_avg = calculate_category_averages(outputs)

    # Handle checkpoint-specific logic
    if sum(outputs.values()) != 0:
        if "checkpoint" in model_name:
            # model_name = model_name.replace("checkpoint", "iter")
            count += 1

            if count == CHECKPOINT_AVERAGE_COUNT:
                # Update cache with current outputs before calculating average
                cache.extend(outputs.values())
                avg = calculate_average_format(cache)
                # Original table row: just model name and individual metrics
                original_row = [model_name] + list(outputs.values())
                # Averages table row: model name, avg, and category averages
                averages_row = [model_name, avg] + category_avg
                return [original_row], [averages_row], [], 0
            else:
                # Update cache with current outputs for accumulation
                cache.extend(outputs.values())
                # Original table row: just model name and individual metrics
                original_row = [model_name] + list(outputs.values())
                # Averages table row: model name, "x", and category averages
                averages_row = [model_name, "x"] + category_avg
                return [original_row], [averages_row], cache, count
        else:
            # For non-checkpoint models, calculate average from current outputs
            avg = calculate_average_format(list(outputs.values()))
            # Original table row: just model name and individual metrics
            original_row = [model_name] + list(outputs.values())
            # Averages table row: model name, avg, and category averages
            averages_row = [model_name, avg] + category_avg
            return [original_row], [averages_row], [], count

    return [], [], cache, count


def process_model_results(model_paths: List[str]) -> Tuple[List[List[Any]], List[List[Any]]]:
    """Process evaluation results for multiple models.

    Args:
        model_paths: List of model directory paths

    Returns:
        Tuple of (original_rows, averages_rows) containing processed results
    """
    original_rows = []
    averages_rows = []
    cache = []
    count = 0

    for model_path in model_paths:
        model_original_rows, model_averages_rows, cache, count = process_single_model(model_path, cache, count)
        original_rows.extend(model_original_rows)
        averages_rows.extend(model_averages_rows)

    return original_rows, averages_rows


# Main Functions
# ==============

def resolve_model_name(alias: str) -> str:
    """Resolve model name alias to full model name.

    Args:
        alias: Model name alias

    Returns:
        Full model name
    """
    return MODEL_NAME_ALIASES.get(alias, alias)


def get_checkpoint_directories(model_names: str | list[str]) -> List[str]:
    """Get sorted list of checkpoint directories for a model.

    Args:
        model_names: Name of the model(s)

    Returns:
        List of checkpoint directory paths
    """
    checkpoint_dirs = []
    if isinstance(model_names, list):
        for model_name in model_names:
            checkpoint_dir = f"{WORKSPACE_CHECKPOINT_DIR}/{model_name}/checkpoints"

            if not os.path.exists(checkpoint_dir):
                print(f"Warning: Checkpoint directory not found: {checkpoint_dir}")
                return []

            
            for item in os.listdir(checkpoint_dir):
                if "checkpoint" in item:
                    checkpoint_dirs.append(os.path.join(checkpoint_dir, item))

        return sorted(checkpoint_dirs)
    else:
        print(f"Processing model: {model_names}")
        checkpoint_dir = f"{WORKSPACE_CHECKPOINT_DIR}/{model_names}/checkpoints"

        if not os.path.exists(checkpoint_dir):
            print(f"Warning: Checkpoint directory not found: {checkpoint_dir}")
            return []

        for item in os.listdir(checkpoint_dir):
            if "checkpoint" in item:
                checkpoint_dirs.append(os.path.join(checkpoint_dir, item))

        return sorted(checkpoint_dirs)

def generate_original_table_headers() -> List[str]:
    """Generate table headers for the original metrics display.

    Returns:
        List of header strings for individual metrics
    """
    base_headers = ["MODEL"]
    
    metric_headers = [
        METRIC_DISPLAY_ALIASES.get(metric, metric)
        for metric in METRICS_CONFIG
    ]

    return base_headers + metric_headers


def generate_averages_table_headers() -> List[str]:
    """Generate table headers for the averages display.

    Returns:
        List of header strings for averages
    """
    return [
        "MODEL",
        "AVG",
        "GEN_AVG",
        "MC_AVG",
        "ENGLISH_AVG",
        "MATH_AVG",
        "CODE_AVG",
        "SCIENCE_AVG"
    ]

def display_results(baseline_original_rows: List[List[Any]], baseline_averages_rows: List[List[Any]], 
                   test_original_rows: List[List[Any]], test_averages_rows: List[List[Any]]) -> None:
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
    public_baseline_original_rows.sort(key=lambda x: float(baseline_averages_rows[baseline_original_rows.index(x)][1]) 
                                      if baseline_averages_rows[baseline_original_rows.index(x)][1] != 'x' else 0, reverse=True)
    public_baseline_averages_rows.sort(key=lambda x: float(x[1]) if x[1] != 'x' else 0, reverse=True)

    # Sort test rows by average score
    sorted_test_averages_rows = sorted(
        test_averages_rows,
        key=lambda x: float(x[1]) if x[1] != 'x' else 0, 
        reverse=True
    )
    # Sort original rows to match the order of averages rows
    sorted_test_original_rows = []
    for avg_row in sorted_test_averages_rows:
        model_name = avg_row[0]
        # Find matching original row
        for orig_row in test_original_rows:
            if orig_row[0] == model_name:
                sorted_test_original_rows.append(orig_row)
                break

    # Combine all rows for each table
    all_original_rows = public_baseline_original_rows + baseline_original_rows[-1:] + sorted_test_original_rows
    all_averages_rows = public_baseline_averages_rows + baseline_averages_rows[-1:] + sorted_test_averages_rows

    # Generate headers
    original_headers = generate_original_table_headers()
    averages_headers = generate_averages_table_headers()

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

    # Get checkpoint directories
    checkpoint_dirs = get_checkpoint_directories(full_model_names)

    # Process baseline models
    baseline_original_rows, baseline_averages_rows = process_model_results(list(BASELINE_MODELS.keys()))

    # Process k2+ model checkpoints
    test_original_rows, test_averages_rows = process_model_results(checkpoint_dirs)

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
