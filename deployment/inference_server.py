import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import logging
from typing import Dict, List

class InferenceServer:
    def __init__(self, base_model_name: str, peft_model_path: str = None):
        self.logger = logging.getLogger(__name__)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        # Load PEFT model if provided
        if peft_model_path:
            self.model = PeftModel.from_pretrained(self.model, peft_model_path)
            self.logger.info(f"Loaded PEFT model from {peft_model_path}")
        
        self.model.eval()
        self.logger.info("Inference server initialized")

    def generate_response(self, prompt: str, max_length: int = 512, 
                         temperature: float = 0.7, top_p: float = 0.9) -> str:
        """Generate a response for the given prompt."""
        try:
            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove the input prompt from the response
            response = response[len(prompt):].strip()
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error generating response: {str(e)}")
            raise

    def answer_question(self, question: str, context: str = "") -> str:
        """Answer a question given optional context."""
        if context:
            prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
        else:
            prompt = f"Question: {question}\n\nAnswer:"
        
        return self.generate_response(prompt)

    def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate responses for multiple prompts."""
        responses = []
        for prompt in prompts:
            response = self.generate_response(prompt, **kwargs)
            responses.append(response)
        return responses

