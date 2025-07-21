from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import Trainer, TrainingArguments

from dataset import SFTDataset

if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained("D:\Qwen2.5-3B-Instruct")
    model = AutoModelForCausalLM.from_pretrained("D:\Qwen2.5-3B-Instruct")
    lora_config = LoraConfig(
        inference_mode=False,
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.1,
        task_type=TaskType.CAUSAL_LM)
    model = get_peft_model(model, lora_config)
    print(model.print_trainable_parameters())
    args = TrainingArguments(output_dir='lora_3B',
                             do_train=True,
                             do_eval=True,
                             seed=42,
                             per_device_train_batch_size=8,
                             per_device_eval_batch_size=10,
                             gradient_accumulation_steps=1,
                             gradient_checkpointing=False,
                             num_train_epochs=1,
                             learning_rate=1e-4,
                             warmup_ratio=0.03,
                             weight_decay=0.1,
                             lr_scheduler_type="cosine",
                             save_strategy="steps",
                             save_steps=100,
                             save_total_limit=3,
                             eval_strategy="steps",
                             eval_steps=100,
                             logging_steps=10,
                             bf16=True)
    train_set = SFTDataset('train.jsonl', tokenizer=tokenizer, prompt_max_seq_len=30, answer_max_seq_len=20)
    val_set = SFTDataset('eval.jsonl', tokenizer=tokenizer, prompt_max_seq_len=30, answer_max_seq_len=20)
    trainer = Trainer(model=model,
                      args=args,
                      train_dataset=train_set,
                      eval_dataset=val_set,
                      tokenizer=tokenizer)
    trainer.train(resume_from_checkpoint=False)


 # https://swanlab.cn/@wangdayaya/Distillation/runs/h331r8b44pu5lf5fui827/chart