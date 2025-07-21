import json
import torch
import torch.utils.checkpoint
from torch.utils.data import Dataset

SYSTEM = "你是一个有帮助的助手，按照格式帮我解决下面数学问题"
class SFTDataset(Dataset):
    def __init__(self, data_path, tokenizer, prompt_max_seq_len, answer_max_seq_len):
        super().__init__()
        self.data = []
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.prompt_max_seq_len = prompt_max_seq_len
        self.answer_max_seq_len = answer_max_seq_len
        self.padding_id = tokenizer.pad_token_id
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                self.data.append(json.loads(line.strip()))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        line = self.data[index]
        instruction_text = line['instruction']
        input_text = line['input']
        output_text = line['output']
        query = instruction_text + input_text
        answer = output_text + self.tokenizer.eos_token
        messages = [{"role": "system", "content": SYSTEM},
                    {'role': 'user', 'content': query}]
        prompt = self.tokenizer.apply_chat_template(messages,  tokenize=False, add_generation_prompt=True)

        instruction = self.tokenizer(prompt, add_special_tokens=False, max_length=self.prompt_max_seq_len,
                                     padding="max_length", pad_to_max_length=True, truncation=True)
        response = self.tokenizer(answer, add_special_tokens=False, max_length=self.answer_max_seq_len,
                                  padding="max_length", pad_to_max_length=True, truncation=True)

        input_ids = instruction["input_ids"] + response["input_ids"]
        attention_mask = instruction["attention_mask"] + response["attention_mask"]
        labels = [-100] * len(instruction["input_ids"]) + response["input_ids"]

        return {'input_ids': torch.tensor(input_ids),
                'attention_mask': torch.tensor(attention_mask),
                'labels': torch.tensor(labels)}
