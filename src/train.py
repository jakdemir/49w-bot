"""
Fine-tunes a base model on 49W transcripts using QLoRA (4-bit quantization + LoRA adapters).

Recommended base model: mistralai/Mistral-7B-Instruct-v0.3 or meta-llama/Llama-3.1-8B-Instruct

Usage:
    python src/train.py --data data/processed/ --output models/49w-v1
    python src/train.py --data data/processed/ --base-model mistralai/Mistral-7B-Instruct-v0.3
"""

import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM


DEFAULT_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
MAX_SEQ_LENGTH = 2048


def load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def format_messages(example: dict, tokenizer) -> dict:
    """Apply chat template to a messages example."""
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}


def train(
    data_dir: str,
    output_dir: str,
    base_model: str = DEFAULT_MODEL,
    epochs: int = 3,
    batch_size: int = 2,
    grad_accum: int = 8,
    lr: float = 2e-4,
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load data
    train_data = load_jsonl(f"{data_dir}/train.jsonl")
    val_data = load_jsonl(f"{data_dir}/val.jsonl")
    print(f"Train: {len(train_data)} examples, Val: {len(val_data)} examples")

    # Quantization config (4-bit QLoRA)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # Load model + tokenizer
    print(f"Loading base model: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    # LoRA config
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Format datasets with chat template
    def format_example(ex):
        return format_messages(ex, tokenizer)

    train_dataset = Dataset.from_list(train_data).map(format_example)
    val_dataset = Dataset.from_list(val_data).map(format_example)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_path),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        bf16=True,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        load_best_model_at_end=True,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        args=training_args,
    )

    print("Starting training...")
    trainer.train()

    print(f"Saving model to {output_path}")
    trainer.save_model(str(output_path))
    tokenizer.save_pretrained(str(output_path))
    print("Training complete.")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune model on 49W transcripts")
    parser.add_argument("--data", default="data/processed", help="Directory with train.jsonl and val.jsonl")
    parser.add_argument("--output", default="models/49w-v1", help="Output directory for trained model")
    parser.add_argument("--base-model", default=DEFAULT_MODEL, help="HuggingFace base model ID")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    args = parser.parse_args()

    train(
        data_dir=args.data,
        output_dir=args.output,
        base_model=args.base_model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        lr=args.lr,
    )


if __name__ == "__main__":
    main()
