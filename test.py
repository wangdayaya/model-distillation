import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import dataset

data = []
with open("eval.jsonl", "r", encoding="utf-8") as f:
    for line in f.readlines():
        data.append(json.loads(line.strip()))
acc = 0
batch_raw_text = []
targets = []

# 原始模型
# model = AutoModelForCausalLM.from_pretrained("D:\Qwen2.5-3B-Instruct", torch_dtype="auto", device_map="auto")
# tokenizer = AutoTokenizer.from_pretrained("D:\Qwen2.5-3B-Instruct")


# 3B-lora-merge 模型
# model = AutoModelForCausalLM.from_pretrained(r"D:\PycharmProjects\Distillation\3B_lora_merge", torch_dtype="auto",
#                                              device_map="auto")
# tokenizer = AutoTokenizer.from_pretrained(r"D:\PycharmProjects\Distillation\3B_lora_merge")


# 3B-Distill-0.5B 模型
model = AutoModelForCausalLM.from_pretrained(r"D:\PycharmProjects\Distillation\3B-Distill-0.5B\checkpoint-338", torch_dtype="auto",
                                             device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(r"D:\PycharmProjects\Distillation\3B-Distill-0.5B\checkpoint-338")

for body in data:
    messages = [
        {"role": "system", "content": dataset.SYSTEM},
        {"role": "user", "content": body["instruction"]}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    batch_raw_text.append(text)
    targets.append(int(body["instruction"]))

batch_input_ids = tokenizer(batch_raw_text, padding='longest')
batch_input_ids = torch.LongTensor(batch_input_ids['input_ids']).to(model.device)

generated_ids = model.generate(batch_input_ids, max_new_tokens=30)
generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(batch_input_ids, generated_ids)]
responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
for i, (t, r) in enumerate(zip(targets, responses)):
    try:
        nums = r.split("--->")[1].split("+")
        p = sum(map(int, nums))
        if t == p:
            acc += 1
        else:
            print("生成错误", t, r)
        if i < 10:
            print(t, r)
    except:
        print("异常", t, r)
        continue
print(acc / len(data))

# 3B 原始模型
# 2710 您提供的数字2710似乎是一串数字，但没有明确的数学问题或运算符。如果您需要对这个数字进行某种计算
# 2711 您提供的数字2711似乎是一个数值，但并没有给出具体的问题。请问您是想要对这个数字进行某种运算，还是说
# 2712 您提供的数字2712是一个具体的数值，但并没有给出任何问题或需要进行的操作。如果您是想询问有关这个数字的问题，
# 2713 您提供的数字2713是一个具体的数值，它本身并不构成一个明确的数学问题。如果您需要对这个数字进行某种操作或
# 2714 您提供的数字2714似乎是一串数字，而不是一个完整的数学问题。如果您需要对这个数字进行某种操作或解决某个与
# 2715 您提供的数字2715是一个具体的数值，它本身并没有一个明确的问题需要解决。不过，如果您有特定的问题或求解的目标
# 2716 您似乎提供了一个数值，但是没有给出具体的问题。2716是一个数字，它可以是任何数学问题的答案，比如一个乘法
# 2717 您提供的数字2717似乎是一个数值，但没有给出具体的问题情境。因此，我无法确定您需要什么样的解答。如果您能
# ...
# 0.0


# 3B-lora-merge 模型

# 2710 王大丫丫的回答是--->694 + 2016
# 2711 王大丫丫的回答是--->2039 + 672
# 2712 王大丫丫的回答是--->586 + 2126
# 2713 王大丫丫的回答是--->2689 + 24
# 2714 王大丫丫的回答是--->205 + 2509
# 2715 王大丫丫的回答是--->490 + 2225
# 2716 王大丫丫的回答是--->380 + 2336
# 2717 王大丫丫的回答是--->183 + 2534
# 2718 王大丫丫的回答是--->439 + 2279
# 2719 王大丫丫的回答是--->206 + 2513
# ...
# 1.0


# 3B-Distill-0.5B 模型
# 2710 王大丫丫的回答是--->120 + 2690
# 2711 王大丫丫的回答是--->1044 + 1667
# 2712 王大丫丫的回答是--->1208 + 1504
# 2713 王大丫丫的回答是--->1211 + 1502
# 2714 王大丫丫的回答是--->104 + 2600
# 2715 王大丫丫的回答是--->108 + 2607
# 2716 王大丫丫的回答是--->120 + 2696
# 2717 王大丫丫的回答是--->111 + 2606
# 2718 王大丫丫的回答是--->120 + 2698
# 2719 王大丫丫的回答是--->1214 + 1505
# 生成错误 2714 王大丫丫的回答是--->104 + 2600
# 生成错误 2710 王大丫丫的回答是--->120 + 2690
# 生成错误 2716 王大丫丫的回答是--->120 + 2696
# 生成错误 2718 王大丫丫的回答是--->120 + 2698
# 生成错误 2724 王大丫丫的回答是--->104 + 2610
# 生成错误 2823 王大丫丫的回答是--->1227 + 1606
# 生成错误 2881 王大丫丫的回答是--->121 + 2750
# 生成错误 2905 王大丫丫的回答是--->108 + 2807
...
# 0.9724137931034482