import logging
from typing import List, Dict
import evaluate
from rouge_score import rouge_scorer
from sacrebleu import corpus_bleu
import time

class Evaluator:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    def evaluate_rouge(self, predictions: List[str], references: List[str]) -> Dict:
        """Evaluates ROUGE scores for predictions vs references."""
        rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
        
        for pred, ref in zip(predictions, references):
            scores = self.rouge_scorer.score(ref, pred)
            rouge_scores['rouge1'].append(scores['rouge1'].fmeasure)
            rouge_scores['rouge2'].append(scores['rouge2'].fmeasure)
            rouge_scores['rougeL'].append(scores['rougeL'].fmeasure)
        
        # Calculate average scores
        avg_scores = {
            'rouge1': sum(rouge_scores['rouge1']) / len(rouge_scores['rouge1']),
            'rouge2': sum(rouge_scores['rouge2']) / len(rouge_scores['rouge2']),
            'rougeL': sum(rouge_scores['rougeL']) / len(rouge_scores['rougeL'])
        }
        
        self.logger.info(f"ROUGE scores: {avg_scores}")
        return avg_scores

    def evaluate_bleu(self, predictions: List[str], references: List[List[str]]) -> float:
        """Evaluates BLEU score for predictions vs references."""
        bleu_score = corpus_bleu(predictions, references)
        self.logger.info(f"BLEU score: {bleu_score.score}")
        return bleu_score.score

    def measure_inference_latency(self, model, tokenizer, questions: List[str], num_runs: int = 10) -> Dict:
        """Measures inference latency and throughput."""
        latencies = []
        
        for _ in range(num_runs):
            start_time = time.time()
            
            for question in questions:
                inputs = tokenizer(question, return_tensors="pt")
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_length=100, do_sample=False)
                
            end_time = time.time()
            latencies.append(end_time - start_time)
        
        avg_latency = sum(latencies) / len(latencies)
        throughput = len(questions) / avg_latency  # questions per second
        
        metrics = {
            'avg_latency_seconds': avg_latency,
            'throughput_qps': throughput
        }
        
        self.logger.info(f"Inference metrics: {metrics}")
        return metrics

    def compare_with_baseline(self, fine_tuned_scores: Dict, baseline_scores: Dict) -> Dict:
        """Compares fine-tuned model performance with baseline."""
        comparison = {}
        
        for metric in fine_tuned_scores:
            if metric in baseline_scores:
                improvement = fine_tuned_scores[metric] - baseline_scores[metric]
                improvement_pct = (improvement / baseline_scores[metric]) * 100
                comparison[f"{metric}_improvement"] = improvement
                comparison[f"{metric}_improvement_pct"] = improvement_pct
        
        self.logger.info(f"Performance comparison: {comparison}")
        return comparison

