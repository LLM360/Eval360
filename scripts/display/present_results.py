import os
import fire
import glob
import json
import statistics
from tabulate import tabulate

# CKPT_DIR = '/lustrefs/users/runner/checkpoints/huggingface'
CKPT_DIR = '/lustrefs/users/runner/checkpoints/huggingface/vocab_trimmed'
METRICS = {
    "arc_challenge": ["english", "mc"],
    "gsm8k": ["math", "gen"],
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
    else:
        assert metric == 'gsm8k'
        return results[metric]['exact_match,flexible-extract']


def read_result(model_dirs):
    cache, rows, count = [], [], 0
    for model_name in model_dirs:
        results, english_sum, math_sum, code_sum, gen_sum, mc_sum, category_avg = {}, [], [], [], [], [], []
        category_sums = [gen_sum, mc_sum, english_sum, math_sum, code_sum]

        for result_file in glob.glob(
                f'{CKPT_DIR}/{model_name}/eval_results/*/*/results_*.json'):
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
        cache.extend(outputs.values())
        for x in category_sums:
            category_avg.append(calc_mean(x))
        if sum(outputs.values()) != 0:
            if "iter" in model_name:
                count += 1
                step = int(model_name.split("_")[-1])
                if count == 4:
                    avg = calc_avg_format(cache)
                    rows.append([model_name, avg] + category_avg + list(outputs.values()))
                    cache = []
                    count = 0
                else:
                    rows.append([model_name, "x"] + category_avg + list(outputs.values()))
            elif cache:
                    avg = calc_avg_format(cache)
                    rows.append([model_name, avg] + category_avg + list(outputs.values()))
                    cache = []
    return rows


def main():
    public_model_dirs, k2_plus_ckpt_dirs = [], []
    for model_name in os.listdir(CKPT_DIR):
        if "iter" in model_name:
            k2_plus_ckpt_dirs.append(model_name)
        elif "bak" not in model_name:
            public_model_dirs.append(model_name)
    # print(public_model_dirs, k2_plus_ckpt_dirs)
    k2_plus_ckpt_dirs = sorted(k2_plus_ckpt_dirs)
    public_rows = read_result(public_model_dirs)
    # sort by avg, desc
    public_rows.sort(key=lambda x: x[1], reverse=True)
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
    print(tabulate(public_rows + sorted(k2_plus_rows, reverse=True), headers=headers, tablefmt="tsv", numalign="right", floatfmt=".2f", maxcolwidths=20))
    

if __name__ == '__main__':
    fire.Fire(main)