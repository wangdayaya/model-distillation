import torch
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import Trainer, TrainingArguments

from dataset import SFTDataset


def compute_fkl(logits, teacher_logits, target, padding_id, reduction="sum", temp=1.0):
    logits = logits / temp
    teacher_logits = teacher_logits / temp
    log_probs = torch.log_softmax(logits, -1, dtype=torch.float32)
    teacher_probs = torch.softmax(teacher_logits, -1, dtype=torch.float32)
    teacher_log_probs = torch.log_softmax(teacher_logits, -1, dtype=torch.float32)
    kl = (teacher_probs * (teacher_log_probs - log_probs))
    kl = kl.sum(-1)
    if reduction == "sum":
        pad_mask = target.eq(padding_id)
        kl = kl.masked_fill_(pad_mask, 0.0)
        kl = kl.sum()
    return kl


class KGTrainer(Trainer):
    def __init__(
            self,
            model=None,
            teacher_model=None,
            if_use_entropy=False,
            args=None,
            data_collator=None,
            train_dataset=None,
            eval_dataset=None,
            tokenizer=None,
            model_init=None,
            compute_metrics=None,
            callbacks=None,
            preprocess_logits_for_metrics=None,
    ):
        super().__init__(
            model,
            args,
            data_collator,
            train_dataset,
            eval_dataset,
            tokenizer,
            model_init,
            compute_metrics,
            callbacks,
            preprocess_logits_for_metrics,
        )
        self.teacher_model = teacher_model
        self.if_use_entropy = if_use_entropy

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        with torch.no_grad():
            teacher_outputs = self.teacher_model(**inputs)
        loss = outputs.loss
        logits = outputs.logits
        teacher_logits = teacher_outputs.logits
        if logits.shape[-1] != teacher_logits.shape[-1]:
            # gap = teacher_logits.shape[-1] - logits.shape[-1]
            # if gap > 0:
            #     pad_logits = torch.zeros((logits.shape[0], logits.shape[1], gap)).to(logits.device)
            #     logits = torch.cat([logits, pad_logits], dim=-1)
            teacher_logits = teacher_logits[:, :, :logits.shape[-1]]
        labels = inputs['labels']
        kl = compute_fkl(logits, teacher_logits, labels, padding_id=-100, temp=2.0)
        if self.if_use_entropy:  # 不微调学生模型的时候，kl 散度损失和交叉熵损失加权
            loss_total = 0.5 * kl + 0.5 * loss
        else:
            loss_total = kl
        return (loss_total, outputs) if return_outputs else loss_total


if __name__ == '__main__':
    # 学生模型
    model = AutoModelForCausalLM.from_pretrained("D:\Qwen2.5-0.5B-Instruct")
    tokenizer = AutoTokenizer.from_pretrained("D:\Qwen2.5-0.5B-Instruct")
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        inference_mode=False,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.1,
        task_type=TaskType.CAUSAL_LM)
    # 使用lora方法训练
    model = get_peft_model(model, lora_config)
    model.cuda()
    print(model.print_trainable_parameters())

    teacher_model = AutoModelForCausalLM.from_pretrained("3B_lora_merge")
    teacher_model.cuda()
    teacher_model.eval()

    args = TrainingArguments(output_dir='3B-Distill-0.5B',
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
    trainer = KGTrainer(model=model,
                        teacher_model=teacher_model,
                        if_use_entropy=True,
                        args=args,
                        train_dataset=train_set,
                        eval_dataset=val_set,
                        tokenizer=tokenizer)
    # 如果是初次训练resume_from_checkpoint为false，接着checkpoint继续训练，为True
    trainer.train(resume_from_checkpoint=False)

# 直接kl 散度蒸馏 https://swanlab.cn/@wangdayaya/Distillation/runs/bj4yjem2w5fpap6xh45xh/chart