import json

import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer


def get_token_distribution(file_path, tokenizer):
    input_num_tokens, outout_num_tokens = [], []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in tqdm(file.readlines()):
            data = json.loads(line)
            text = data["instruction"]
            label = data["output"]
            label = json.dumps(label, ensure_ascii=False)
            messages = [
                {"role": "system", "content": "你是一个有帮助的助手，模仿帮我解决下面数学问题"},
                {"role": "user", "content": text}
            ]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            instruction = tokenizer(prompt, )
            input_num_tokens.append(len(instruction["input_ids"]))

            response = tokenizer(label, )
            outout_num_tokens.append(len(response["input_ids"]))
    return min(input_num_tokens), max(input_num_tokens), np.mean(input_num_tokens), np.percentile(input_num_tokens, 95), \
            min(outout_num_tokens), max(outout_num_tokens), np.mean(outout_num_tokens), np.percentile(outout_num_tokens, 95),


def main():
    model_path = "D:\Qwen2.5-3B-Instruct"
    train_data_path = r"train.jsonl"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    i_min, i_max, i_avg, i_95, o_min, o_max, o_avg, o_95 = get_token_distribution(train_data_path, tokenizer)
    print(
        f"i_min：{i_min}, i_max：{i_max}, i_avg：{i_avg}, i_95:{i_95}, o_min：{o_min}, o_max：{o_max}, o_avg：{o_avg}, o_95:{o_95}")


main()