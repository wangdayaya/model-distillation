from transformers import AutoModelForCausalLM
from peft import PeftModel
from transformers import AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(r"D:\Qwen2.5-3B-Instruct")
model = PeftModel.from_pretrained(model, r"D:\PycharmProjects\Distillation\lora_3B\checkpoint-338")
merged_model = model.merge_and_unload()

# 把合并后的模型保存到指定的目录
merged_model.save_pretrained("3B_lora_merge", max_shard_size="2048MB", safe_serialization=True)
tokenizer = AutoTokenizer.from_pretrained(r"D:\Qwen2.5-3B-Instruct")
tokenizer.save_pretrained("3B_lora_merge")
