import logging
from typing import List, Dict
from datasets import Dataset
import random

class BenchmarkGenerator:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def generate_benchmark_dataset(self, qa_pairs: List[Dict], num_samples: int = 100) -> Dataset:
        if len(qa_pairs) < num_samples:
            self.logger.warning(f"Not enough QA pairs ({len(qa_pairs)}) to generate {num_samples} benchmark samples. Using all available.")
            samples = qa_pairs
        else:
            samples = random.sample(qa_pairs, num_samples)

        benchmark_data = []
        for qa in samples:
            benchmark_data.append({
                "question": qa["question"],
                "ground_truth_answer": qa["answer"],
                "context": qa["context"]
            })
        
        self.logger.info(f"Generated benchmark dataset with {len(benchmark_data)} samples.")
        return Dataset.from_list(benchmark_data)


