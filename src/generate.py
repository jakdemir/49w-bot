"""
Generate video content given a title, using the fine-tuned 49W model.

Usage:
    python src/generate.py --model models/49w-v1 --title "Yapay Zeka ile Para Kazanmak"
    python src/generate.py --model models/49w-v1 --title "..." --max-tokens 1000
"""

import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


SYSTEM_PROMPT = (
    "Sen 49W YouTube kanalındaki bir konuşmacısın. "
    "Verilen başlığa göre, kanaldaki video içeriğini o kişilerin üslubu ve bilgisiyle üret."
)


def load_model(model_path: str, base_model: str | None = None):
    """Load the fine-tuned model. Handles both merged and LoRA adapter formats."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if base_model:
        # Load as LoRA adapter on top of base model
        print(f"Loading base model: {base_model}")
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=bnb_config,
            device_map="auto",
        )
        print(f"Loading LoRA adapters from: {model_path}")
        model = PeftModel.from_pretrained(model, model_path)
    else:
        # Load as a merged / standalone model
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
        )

    model.eval()
    return model, tokenizer


def generate(
    model,
    tokenizer,
    title: str,
    max_new_tokens: int = 800,
    temperature: float = 0.8,
    top_p: float = 0.95,
    repetition_penalty: float = 1.1,
) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Başlık: {title}"},
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser(description="Generate 49W-style content from a title")
    parser.add_argument("--model", default="models/49w-v1", help="Path to fine-tuned model or adapter directory")
    parser.add_argument("--base-model", default=None, help="Base model ID if --model is a LoRA adapter")
    parser.add_argument("--title", required=True, help="Video title to generate content for")
    parser.add_argument("--max-tokens", type=int, default=800, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    args = parser.parse_args()

    print(f"Loading model from: {args.model}")
    model, tokenizer = load_model(args.model, args.base_model)

    print(f"\nGenerating for title: \"{args.title}\"\n")
    print("=" * 60)
    result = generate(model, tokenizer, args.title, args.max_tokens, args.temperature, args.top_p)
    print(result)
    print("=" * 60)


if __name__ == "__main__":
    main()
