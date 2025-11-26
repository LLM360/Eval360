"""
Shared utilities for presenting evaluation results.
This module contains all common logic used by both base and instruct result presentation scripts.
"""
import os
import glob
import json
import re
import statistics
from typing import Dict, List, Tuple, Any, Union, Callable

# Base checkpoint directories
BASE_CHECKPOINT_DIR = "/lustrefs/users/runner/checkpoints/huggingface"
WORKSPACE_CHECKPOINT_DIR = "/lustrefs/users/runner/workspace/checkpoints/huggingface"

# Result extraction keys for different metrics (merged from base and instruct)
RESULT_EXTRACTION_KEYS = {
    'arc_challenge': 'acc_norm,none',
    'hellaswag': 'acc_norm,none',
    'leaderboard_gpqa_diamond': 'acc_norm,none',
    'gpqa_diamond_cot_zeroshot': 'exact_match,flexible-extract',
    'gpqa_diamond_reasoning_base': 'exact_match,none',
    'gpqa_diamond_reasoning_instruct': 'exact_match,none',
    'piqa': 'acc_norm,none',
    'mmlu': 'acc,none',
    'truthfulqa_mc2': 'acc,none',
    'winogrande': 'acc,none',
    'mmlu_arabic': 'acc,none',
    'mmlu_pro': 'exact_match,custom-extract',
    'bbh': 'exact_match,get-answer',
    'mbpp': 'pass_at_1,none',
    'mbpp_instruct': 'pass_at_1,extract_code',
    'humaneval': 'pass@1,create_test',
    'humaneval_64': 'pass@64,create_test',
    'humaneval_instruct': 'pass@1,create_test',
    'humaneval_64_instruct': 'pass@64,create_test',
    'gsm8k_cot': 'exact_match,flexible-extract',
    'gsm8k': 'exact_match,flexible-extract',
    'gsm8k_reasoning_base': 'math_verify,none',
    'gsm8k_reasoning_instruct': 'math_verify,none',
    'minerva_math': 'math_verify,none',
    'minerva_math_reasoning_base': 'math_verify,none',
    'minerva_math_reasoning_instruct': 'math_verify,none',
    'minerva_math500': 'math_verify,none',
    'aime24': 'exact_match,none',
    'aime25': 'exact_match,none'
}

# Metric name aliases for better display (merged from base and instruct)
METRIC_DISPLAY_ALIASES = {
    'arc_challenge': 'ARC-C',
    'hellaswag': 'HellaSwag',
    'leaderboard_gpqa_diamond': 'GPQA-Diamond-MC',
    'gpqa_diamond_cot_zeroshot': 'GPQA-Diamond-CoT',
    'gpqa_diamond_reasoning_base': 'GPQA-Diamond-Reasoning',
    'gpqa_diamond_reasoning_instruct': 'GPQA-Diamond-Reasoning',
    'piqa': 'PIQA',
    'mmlu': 'MMLU',
    'truthfulqa_mc2': 'TruthfulQA',
    'winogrande': 'WinoGrande',
    'mmlu_arabic': 'MMLU-Arabic',
    'mmlu_pro': 'MMLU-Pro',
    'bbh': 'BBH',
    'mbpp': 'MBPP',
    'mbpp_instruct': 'MBPP-Instruct',
    'humaneval': 'HumanEval',
    'humaneval_64': 'HumanEval-64',
    'humaneval_instruct': 'HumanEval',
    'humaneval_64_instruct': 'HumanEval-64',
    'gsm8k_cot': 'GSM8K-CoT',
    'gsm8k': 'GSM8K',
    'gsm8k_reasoning_base': 'GSM8K-Reasoning',
    'gsm8k_reasoning_instruct': 'GSM8K-Reasoning',
    'minerva_math': 'Minerva-MATH',
    'minerva_math_reasoning_base': 'Minerva-MATH-Reasoning',
    'minerva_math_reasoning_instruct': 'Minerva-MATH-Reasoning',
    'minerva_math500': 'MATH-500',
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


def extract_result_value(
    results: Dict[str, Any], 
    metric: str, 
    extraction_keys: Dict[str, str]
) -> float:
    """Extract the appropriate result value for a given metric.

    Args:
        results: Dictionary containing metric results
        metric: Name of the metric to extract
        extraction_keys: Dictionary mapping metrics to their extraction keys

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
    extraction_key = extraction_keys.get(metric)
    if extraction_key and extraction_key in results[metric]:
        return results[metric][extraction_key]

    return 0.0


def load_model_results(
    model_path: str, 
    metrics_config: Dict[str, List[str]]
) -> Dict[str, Any]:
    """Load evaluation results for a single model.

    Args:
        model_path: Path to the model directory
        metrics_config: Dictionary mapping metric names to their categories

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
                if key in metrics_config:
                    if key in results:
                        results[key] = {**results[key], **value}
                    else:
                        results[key] = value
        except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
            print(f"Warning: Could not load results from {result_file}: {e}")
            continue

    return results


def calculate_category_averages(
    outputs: Dict[str, float],
    metrics_config: Dict[str, List[str]],
    category_order: List[str]
) -> List[float]:
    """Calculate category averages for different types of tasks.

    Args:
        outputs: Dictionary mapping metric names to their scores
        metrics_config: Dictionary mapping metric names to their categories
        category_order: List of category names in the desired order

    Returns:
        List of averages in the specified category order
    """
    category_sums = {category: [] for category in category_order}

    for metric, score in outputs.items():
        categories = metrics_config.get(metric, [])
        for category in categories:
            if category in category_sums:
                category_sums[category].append(score)

    # Return averages in the expected order
    return [calculate_mean(category_sums[category]) for category in category_order]


def process_single_model(
    model_path: str,
    cache: List[float],
    count: int,
    metrics_config: Dict[str, List[str]],
    extraction_keys: Dict[str, str],
    category_order: List[str],
    baseline_models: Dict[str, str],
    checkpoint_average_count: int,
    get_model_name: Callable[[str], str],
    replace_checkpoint_name: bool = True
) -> Tuple[List, List, List[float], int]:
    """Process results for a single model.

    Args:
        model_path: Path to the model directory
        cache: List of cached values for averaging
        count: Current count for checkpoint averaging
        metrics_config: Dictionary mapping metric names to their categories
        extraction_keys: Dictionary mapping metrics to extraction keys
        category_order: List of category names in desired order
        baseline_models: Dictionary mapping model paths to display names
        checkpoint_average_count: Number of checkpoints to average
        get_model_name: Function to extract model name from path
        replace_checkpoint_name: Whether to replace "checkpoint" with "iter" in model name

    Returns:
        Tuple of (original_row_data, averages_row_data, updated_cache, updated_count)
    """
    model_name = baseline_models.get(model_path, get_model_name(model_path))

    # Load results for this model
    results = load_model_results(model_path, metrics_config)

    # Extract outputs for all metrics
    outputs = {
        metric: extract_result_value(results, metric, extraction_keys) * 100
        for metric in metrics_config
    }

    # Calculate category averages
    category_avg = calculate_category_averages(outputs, metrics_config, category_order)

    # Handle checkpoint-specific logic
    if sum(outputs.values()) != 0:
        if "checkpoint" in model_name:
            if replace_checkpoint_name:
                model_name = model_name.replace("checkpoint", "iter")
            count += 1

            if count == checkpoint_average_count:
                # Update cache with current outputs before calculating average
                cache.extend(outputs.values())
                avg = calculate_average_format(cache)
                original_row = [model_name] + list(outputs.values())
                averages_row = [model_name, avg] + category_avg
                return [original_row], [averages_row], [], 0
            else:
                # Update cache with current outputs for accumulation
                cache.extend(outputs.values())
                original_row = [model_name] + list(outputs.values())
                averages_row = [model_name, "x"] + category_avg
                return [original_row], [averages_row], cache, count
        else:
            # For non-checkpoint models, calculate average from current outputs
            avg = calculate_average_format(list(outputs.values()))
            original_row = [model_name] + list(outputs.values())
            averages_row = [model_name, avg] + category_avg
            return [original_row], [averages_row], [], count

    return [], [], cache, count


def process_model_results(
    model_paths: List[str],
    metrics_config: Dict[str, List[str]],
    extraction_keys: Dict[str, str],
    category_order: List[str],
    baseline_models: Dict[str, str],
    checkpoint_average_count: int,
    get_model_name: Callable[[str], str],
    replace_checkpoint_name: bool = True
) -> Tuple[List[List[Any]], List[List[Any]]]:
    """Process evaluation results for multiple models.

    Args:
        model_paths: List of model directory paths
        metrics_config: Dictionary mapping metric names to their categories
        extraction_keys: Dictionary mapping metrics to extraction keys
        category_order: List of category names in desired order
        baseline_models: Dictionary mapping model paths to display names
        checkpoint_average_count: Number of checkpoints to average
        get_model_name: Function to extract model name from path
        replace_checkpoint_name: Whether to replace "checkpoint" with "iter" in model name

    Returns:
        Tuple of (original_rows, averages_rows) containing processed results
    """
    original_rows = []
    averages_rows = []
    cache = []
    count = 0

    for model_path in model_paths:
        model_original_rows, model_averages_rows, cache, count = process_single_model(
            model_path, cache, count, metrics_config, extraction_keys, category_order,
            baseline_models, checkpoint_average_count, get_model_name, replace_checkpoint_name
        )
        original_rows.extend(model_original_rows)
        averages_rows.extend(model_averages_rows)

    return original_rows, averages_rows


def get_checkpoint_directories(
    model_names: Union[str, List[str]],
    workspace_checkpoint_dir: str = WORKSPACE_CHECKPOINT_DIR
) -> List[str]:
    """Get sorted list of checkpoint directories for a model or models.

    Args:
        model_names: Name(s) of the model(s) - can be a string or list of strings
        workspace_checkpoint_dir: Base directory for workspace checkpoints

    Returns:
        List of checkpoint directory paths
    """
    checkpoint_dirs = []
    
    if isinstance(model_names, list):
        for model_name in model_names:
            checkpoint_dir = f"{workspace_checkpoint_dir}/{model_name}/checkpoints"
            
            if not os.path.exists(checkpoint_dir):
                print(f"Warning: Checkpoint directory not found: {checkpoint_dir}")
                continue
            
            for item in os.listdir(checkpoint_dir):
                if "checkpoint" in item:
                    checkpoint_dirs.append(os.path.join(checkpoint_dir, item))
    else:
        checkpoint_dir = f"{workspace_checkpoint_dir}/{model_names}/checkpoints"
        
        if not os.path.exists(checkpoint_dir):
            print(f"Warning: Checkpoint directory not found: {checkpoint_dir}")
            return []
        
        for item in os.listdir(checkpoint_dir):
            if "checkpoint" in item:
                checkpoint_dirs.append(os.path.join(checkpoint_dir, item))

    return sorted(checkpoint_dirs)


def generate_original_table_headers(
    metrics_config: Dict[str, List[str]],
    metric_display_aliases: Dict[str, str]
) -> List[str]:
    """Generate table headers for the original metrics display.

    Args:
        metrics_config: Dictionary mapping metric names to their categories
        metric_display_aliases: Dictionary mapping metric names to display names

    Returns:
        List of header strings for individual metrics
    """
    base_headers = ["MODEL"]
    
    metric_headers = [
        metric_display_aliases.get(metric, metric)
        for metric in metrics_config
    ]

    return base_headers + metric_headers


def generate_averages_table_headers(category_order: List[str]) -> List[str]:
    """Generate table headers for the averages display.

    Args:
        category_order: List of category names in desired order

    Returns:
        List of header strings for averages
    """
    category_headers = [f"{category.upper()}_AVG" for category in category_order]
    return ["MODEL", "AVG"] + category_headers


def extract_checkpoint_number(model_name: str) -> int:
    """Extract checkpoint number from model name.
    
    Args:
        model_name: Model name that may contain checkpoint/iter number
        
    Returns:
        Checkpoint number as integer, or 0 if not found
    """
    # Look for patterns like checkpoint_0005000, iter_0005000, etc.
    match = re.search(r'(?:checkpoint|iter)[_-]?(\d+)', model_name, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return 0


def sort_rows_by_checkpoint(
    original_rows: List[List[Any]],
    averages_rows: List[List[Any]]
) -> Tuple[List[List[Any]], List[List[Any]]]:
    """Sort rows by checkpoint number in descending order, matching original and averages rows.

    Args:
        original_rows: List of original metric rows
        averages_rows: List of average rows

    Returns:
        Tuple of (sorted_original_rows, sorted_averages_rows)
    """
    # Create a mapping from model name to original row
    model_to_original = {row[0]: row for row in original_rows}
    
    # Sort averages rows by checkpoint number in descending order
    sorted_averages_rows = sorted(
        averages_rows,
        key=lambda x: extract_checkpoint_number(x[0]),
        reverse=True
    )
    
    # Sort original rows to match the order of averages rows
    sorted_original_rows = []
    for avg_row in sorted_averages_rows:
        model_name = avg_row[0]
        if model_name in model_to_original:
            sorted_original_rows.append(model_to_original[model_name])
    
    return sorted_original_rows, sorted_averages_rows


def sort_rows_by_average(
    original_rows: List[List[Any]],
    averages_rows: List[List[Any]]
) -> Tuple[List[List[Any]], List[List[Any]]]:
    """Sort rows by average score, matching original and averages rows.

    Args:
        original_rows: List of original metric rows
        averages_rows: List of average rows

    Returns:
        Tuple of (sorted_original_rows, sorted_averages_rows)
    """
    # Create a mapping from model name to original row
    model_to_original = {row[0]: row for row in original_rows}
    
    # Sort averages rows by average score
    sorted_averages_rows = sorted(
        averages_rows,
        key=lambda x: float(x[1]) if x[1] != 'x' else 0,
        reverse=True
    )
    
    # Sort original rows to match the order of averages rows
    sorted_original_rows = []
    for avg_row in sorted_averages_rows:
        model_name = avg_row[0]
        if model_name in model_to_original:
            sorted_original_rows.append(model_to_original[model_name])
    
    return sorted_original_rows, sorted_averages_rows

