import fire
from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_NAME = 'opensourcerelease/DeepSeek-V3-Base-bf16'
OUTPUT_DIR = '/lustrefs/users/runner/checkpoints/huggingface/deepseek-v3-base-bf16-new'
HF_TOKEN = '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'


def main():
    # tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
    # model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, token=HF_TOKEN, device_map="auto")
    # tokenizer.save_pretrained(OUTPUT_DIR)
    # model.save_pretrained(OUTPUT_DIR, safe_serialization=False)
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=MODEL_NAME, cache_dir=OUTPUT_DIR)


if __name__ == '__main__':
    fire.Fire(main)