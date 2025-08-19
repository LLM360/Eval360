import os
import fire
import glob
import json
import statistics
from tabulate import tabulate

# CKPT_DIR = '/lustrefs/users/runner/checkpoints/huggingface'
# CKPT_DIR = '/lustrefs/users/runner/checkpoints/huggingface/vocab_trimmed'
METRICS = {
    "arc_challenge": ["english", "mc"],
    "gsm8k": ["math", "gen"],
    "gsm8k_cot": ["math", "gen"],
    "minerva_math": ["math", "gen"],
    "hellaswag": ["english", "mc"],
    "mmlu": ["english", "mc"],
    "mmlu_arabic": ["arabic", "mc"],
    "truthfulqa_mc2": ["english", "mc"],
    "winogrande": ["english", "mc"],
    "leaderboard_gpqa_diamond": ["english", "mc"],
    "bbh": ["english", "gen"],
    "mmlu_pro": ["english", "mc"],
    "mbpp": ["code", "gen"],
    "humaneval": ["code", "gen"],
    "ifeval": ["english"],
    "piqa": ["english", "mc"]
}
WINDOW_INTERVAL = 20000
BASELINE_MODELS = {
    "/lustrefs/users/runner/checkpoints/huggingface/k2-65b": "k2-65b",
    "/lustrefs/users/runner/checkpoints/huggingface/llama3-70b": "llama3-70b",
    "/lustrefs/users/runner/checkpoints/huggingface/qwen2.5-32b": "qwen2.5-32b",
    "/lustrefs/users/runner/checkpoints/huggingface/qwen2.5-72b": "qwen2.5-72b",
    "/lustrefs/users/runner/checkpoints/huggingface/falcon-h1-34b": "falcon-h1-34b",
    "/lustrefs/users/runner/checkpoints/huggingface/llama3.1-70b": "llama3.1-70b",
    # "/lustrefs/users/runner/checkpoints/huggingface/vocab_trimmed/iter_1249000": "pretrained",
    "/lustrefs/users/runner/workspace/checkpoints/huggingface/k2plus_stage1_attn8k_jais250k_tp8/checkpoints/checkpoint_0135000": "midtrain-stage1"
}

def calc_avg_format(value_list):
    if 0 in value_list:
        return '0.00'
    return f'{statistics.mean(value_list):.2f}'


def calc_mean(num_list):
    if not num_list:
        return 0.00
    if 0 in num_list:
        return 0.00
    return statistics.mean(num_list)


def get_result(results, metric):
    if metric not in results:
        return 0
    if metric in ['arc_challenge', 'hellaswag', 'leaderboard_gpqa_diamond', 'piqa']:
        return results[metric]['acc_norm,none']
    elif metric in ['mmlu', 'truthfulqa_mc2', 'winogrande', 'mmlu_arabic']:
        return results[metric]['acc,none']
    elif metric in ['mmlu_pro']:
        return results[metric]['exact_match,custom-extract']
    elif metric in ['bbh']:
        return results[metric]['exact_match,get-answer']
    elif metric in ['mbpp']:
        return results[metric]['pass_at_1,none']
    elif metric in ['humaneval']:
        return results[metric]['pass@1,create_test']
    elif metric in ['ifeval']:
        return statistics.mean([results[metric]['prompt_level_strict_acc,none'], results[metric]['inst_level_strict_acc,none']])
    elif metric in ['gsm8k_cot']:
        return results[metric]['exact_match,flexible-extract']
    elif metric in ['minerva_math']:
        return results[metric]['exact_match,none']
    else:
        assert metric == 'gsm8k'
        return results[metric]['exact_match,flexible-extract']


def read_result(models):
    cache, rows, count = [], [], 0
    for model_path in models:
        model_name = BASELINE_MODELS.get(model_path, model_path.split("/")[-1])
        # model_name = model_path.split('/')[-1]
        results, english_sum, math_sum, code_sum, gen_sum, mc_sum, category_avg = {}, [], [], [], [], [], []
        category_sums = [gen_sum, mc_sum, english_sum, math_sum, code_sum]

        for result_file in glob.glob(
                f'{model_path}/eval_results/*/*/results_*.json'):
            if 'gsm8k_0shots' in result_file:
                continue
            for key, value in json.load(open(result_file))['results'].items():
                if key in METRICS:
                    results[key] = value

        outputs = {metric: get_result(results, metric=metric) * 100 for metric in METRICS}
        for key, output in outputs.items():
            if "english" in METRICS[key]:
                english_sum.append(output)
            if "math" in METRICS[key]:
                math_sum.append(output)
            if "code" in METRICS[key]:
                code_sum.append(output)
            if "gen" in METRICS[key]:
                gen_sum.append(output)
            if "mc" in METRICS[key]:
                mc_sum.append(output)
        # print(model_name, outputs, cache)
        cache.extend(outputs.values())
        for x in category_sums:
            category_avg.append(calc_mean(x))
        if sum(outputs.values()) != 0:
            if "checkpoint" in model_name:
                model_name = model_name.replace("checkpoint", "iter")
                count += 1
                step = int(model_name.split("_")[-1])
                if count == 4:
                    avg = calc_avg_format(cache)
                    rows.append([model_name, avg] + category_avg + list(outputs.values()))
                    cache = []
                    count = 0
                else:
                    rows.append([model_name, "x"] + category_avg + list(outputs.values()))
            else:
                avg = calc_avg_format(cache)
                rows.append([model_name, avg] + category_avg + list(outputs.values()))
                cache = []
    return rows


def main(model_name):
    if model_name == "stage1_v1":
        model_name = "k2plus_data.v1_attn8k_jais250k_tp8"
    elif model_name == "stage1_v2":
        model_name = "k2plus_data.v2_attn8k_jais250k_tp8"
    elif model_name == "stage1_v3":
        model_name = "k2plus_data.v3_attn8k_jais250k_tp8"
    elif model_name == "stage1_v4":
        model_name = "k2plus_data.v4_attn8k_jais250k_tp8"
    elif model_name == "stage1":
        model_name = "k2plus_stage1_attn8k_jais250k_tp8"
    elif model_name == "stage2_v1":
        model_name = "k2plus_stage2_attn64k_jais250k_tp8_normal"
    elif model_name == "stage2_v2":
        model_name = "k2plus_stage2_attn64k_jais250k_tp8_bestfit"
    elif model_name == "stage2":
        model_name = "k2plus_stage2_attn64k_jais250k_tp8_bestfit_fix"
    CKPT_DIR="/lustrefs/users/runner/workspace/checkpoints"
    CKPT_DIR=f"{CKPT_DIR}/huggingface/{model_name}/checkpoints"
    k2_plus_ckpt_dirs = []
    for model_name in os.listdir(CKPT_DIR):
        if "checkpoint" in model_name:
            k2_plus_ckpt_dirs.append(os.path.join(CKPT_DIR, model_name))
    k2_plus_ckpt_dirs = sorted(k2_plus_ckpt_dirs)
    baseline_rows = read_result(BASELINE_MODELS)
    # sort by avg, desc
    public_baseline_rows = baseline_rows[:-1]
    public_baseline_rows.sort(key=lambda x: x[1], reverse=True)
    k2_plus_rows = read_result(k2_plus_ckpt_dirs)
    headers = [
        "model",
        "avg",
        "gen_avg",
        "mc_avg",
        "english_avg", 
        "math_avg",
        "code_avg"
    ] + [metric if metric != "leaderboard_gpqa_diamond" else "gpqa_diamond" for metric in METRICS]
    print(tabulate(public_baseline_rows + baseline_rows[-1:] + sorted(k2_plus_rows, reverse=True), headers=headers, tablefmt="tsv", numalign="right", floatfmt=".2f", maxcolwidths=20))
    

if __name__ == '__main__':
    fire.Fire(main)