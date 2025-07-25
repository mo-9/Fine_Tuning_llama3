from typing import List, Dict
from datasets import Dataset

class TrainingDataFormatter:
    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer

    def format_for_sft(self, qa_pairs: List[Dict]) -> List[Dict]:
        formatted_data = []
        for qa in qa_pairs:

            formatted_data.append({
                "instruction": qa["question"],
                "input": qa["context"],
                "output": qa["answer"]
            })
        return formatted_data

    def create_dataset(self, formatted_data: List[Dict]) -> Dataset:
        return Dataset.from_list(formatted_data)

    def tokenize_function(self, examples):
        if not self.tokenizer:
            raise ValueError("Tokenizer must be provided for tokenization.")
        

        texts = [
            f"Instruction: {inst}\nInput: {inp}\nOutput: {out}"
            for inst, inp, out in zip(examples["instruction"], examples["input"], examples["output"])
        ]
        
        return self.tokenizer(texts, truncation=True, padding="max_length")


