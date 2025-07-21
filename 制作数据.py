import json
import random


def generate_numbers(A):
    B = random.randint(0, A - 1)
    C = A - B
    return f"王大丫丫的回答是--->{B} + {C}"


def make_data():
    with open("train.jsonl", "w", encoding="utf-8") as f:
        for i in range(10, 3000):
            d = {"instruction": f"{i}", "input": "", "output": generate_numbers(i)}
            line = json.dumps(d, ensure_ascii=False)
            f.write(line + "\n")

make_data()