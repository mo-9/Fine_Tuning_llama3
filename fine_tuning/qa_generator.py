import logging
from typing import List, Dict
from transformers import pipeline

class QAGenerator:
    def __init__(self, model_name: str = "distilbert-base-cased-distilled-squad"):
        self.qa_pipeline = pipeline("question-answering", model=model_name, tokenizer=model_name)
        self.logger = logging.getLogger(__name__)

    def generate_qa_pairs(self, context: str, num_questions: int = 3) -> List[Dict]:
        qa_pairs = []


        sentences = [s.strip() for s in context.split(".") if s.strip()]
        
        for i, sentence in enumerate(sentences[:num_questions]):
            try:

                question = f"What is {sentence.split( ' ')[0].lower()}?"
                
                answer_result = self.qa_pipeline(question=question, context=context)
                answer = answer_result["answer"]
                
                qa_pairs.append({
                    "question": question,
                    "answer": answer,
                    "context": context
                })
            except Exception as e:
                self.logger.warning(f"Could not generate QA pair for sentence '{sentence}': {e}")
                continue
                
        return qa_pairs

    def generate_qa_from_documents(self, documents: List[Dict], num_questions_per_doc: int = 3) -> List[Dict]:
        """Generates QA pairs from a list of documents."""
        all_qa_pairs = []
        for doc in documents:
            if "content" in doc and doc["content"]:
                qa_pairs = self.generate_qa_pairs(doc["content"], num_questions_per_doc)
                for qa in qa_pairs:
                    qa["source_doc_id"] = doc.get("id")
                    all_qa_pairs.append(qa)
        return all_qa_pairs


