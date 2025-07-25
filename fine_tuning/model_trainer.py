import logging
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
import torch
import os

class ModelTrainer:
    def __init__(self, model_name: str = "meta-llama/Llama-3-7b-hf", output_dir: str = "./results"):
        self.model_name = model_name
        self.output_dir = output_dir
        self.logger = logging.getLogger(__name__)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            load_in_4bit=True,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.model.config.use_cache = False
        self.model.config.pretraining_tp = 1

        self.model = prepare_model_for_kbit_training(self.model)

        self.lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(self.model, self.lora_config)

    def train(self, train_dataset: Dataset, eval_dataset: Dataset = None):
        """Fine-tunes the model using LoRA."""
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=2,
            learning_rate=2e-4,
            logging_steps=10,
            max_steps=50,
            save_steps=10,
            fp16=False, # Set to True if your GPU supports it
            bf16=True,  # Set to True if your GPU supports it
            optim="paged_adamw_8bit",
            logging_dir="./logs",
            report_to="none", # Can be "mlflow", "wandb", etc.
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
        )

        self.logger.info("Starting model training...")
        trainer.train()
        self.logger.info("Training complete.")

        # Save the fine-tuned model
        trainer.save_model(os.path.join(self.output_dir, "final_checkpoint"))
        self.tokenizer.save_pretrained(os.path.join(self.output_dir, "final_checkpoint"))
        self.logger.info(f"Model saved to {os.path.join(self.output_dir, 'final_checkpoint')}")

    def get_tokenizer(self):
        return self.tokenizer

    def get_model(self):
        return self.model


