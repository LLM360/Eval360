import fire
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.mistral3 import Mistral3Model


MODEL_NAME = 'mistralai/Mistral-Small-3.1-24B-Instruct-2503'
OUTPUT_DIR = '/lustrefs/users/runner/checkpoints/huggingface/mistral-small-3.1-24b-instruct-2503'
HF_TOKEN = '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'


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, token=HF_TOKEN, device_map="auto")
    tokenizer.save_pretrained(OUTPUT_DIR)
    model.save_pretrained(OUTPUT_DIR, safe_serialization=False)
    # from huggingface_hub import snapshot_download
    # snapshot_download(repo_id="deepseek-ai/DeepSeek-V3-Base", allow_patterns="inference/*")


if __name__ == '__main__':
    fire.Fire(main)