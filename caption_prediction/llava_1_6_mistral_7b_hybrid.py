"""Hybrid fine-tuning of LLaVA 1.6 Mistral 7B for ImageCLEFmedical 2024 caption prediction.

Model: liuhaotian/llava-v1.6-mistral-7b (LLaVA-NeXT with a Mistral-7B-Instruct language
model and a CLIP ViT-L/336 vision tower), loaded in bfloat16 through the original LLaVA
code base.

"Hybrid" fine-tuning means two things are trained at the same time:
  * LoRA adapters (r=32, rsLoRA-style alpha = 32 * sqrt(32)) on the attention and MLP
    projections of the language model and vision tower, and on both linear layers of the
    multimodal projector (mm_projector.0 / mm_projector.2);
  * the full weights of `lm_head` and `embed_tokens`, which are unfrozen after the LoRA
    wrapper is applied.

The script can:
  * --mode setup     clone the LLaVA repository and patch it (one-time)
  * --mode train     fine-tune on the ROCOv2 train/validation split (HF dataset)
  * --mode predict   generate captions for the unlabeled test set and write a CSV (ID, predicted_caption)
  * --mode evaluate  generate captions for the validation split and write a CSV
                     (correct_caption, predicted_caption) for scoring

Usage:
    python llava_1_6_mistral_7b_hybrid.py --mode setup
    pip install -e LLaVA                      # once, after setup
    python llava_1_6_mistral_7b_hybrid.py --mode train
    python llava_1_6_mistral_7b_hybrid.py --mode predict --adapter-path Models/<run>/checkpoint-XXXX
    python llava_1_6_mistral_7b_hybrid.py --mode evaluate --adapter-path Models/<run>/checkpoint-XXXX

Environment:
    HF_TOKEN       Hugging Face token for the private datasets (or run `huggingface-cli login`).
    WANDB_API_KEY  Optional. `report_to=None` reports to every installed integration, so
                   Weights & Biases is used when it is installed and logged in.

Requirements (versions used in the original run):
    torch==2.1.2, transformers==4.37.2, accelerate==0.21.0, sentencepiece==0.1.99,
    bitsandbytes, peft, datasets, hf_transfer, pandas, Pillow, requests, flash-attn,
    LLaVA (https://github.com/haotian-liu/LLaVA, installed with `pip install -e LLaVA`)
"""

import argparse
import math
import os
import re
import subprocess
import sys
from io import BytesIO

# Must be set before huggingface_hub is imported for faster downloads.
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

import pandas as pd
import requests
import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel, get_peft_model
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from transformers import EarlyStoppingCallback, Trainer, TrainerCallback, TrainingArguments

try:
    from llava.constants import (
        DEFAULT_IM_END_TOKEN,
        DEFAULT_IM_START_TOKEN,
        DEFAULT_IMAGE_TOKEN,
        IMAGE_PLACEHOLDER,
        IMAGE_TOKEN_INDEX,
    )
    from llava.conversation import conv_templates
    from llava.mm_utils import get_model_name_from_path, tokenizer_image_token
    from llava.model.builder import load_pretrained_model
    from llava.utils import disable_torch_init
except ImportError:  # Allowed so that `--mode setup` can run before LLaVA is installed.
    llava_available = False
else:
    llava_available = True


# --- Defaults ---

MODEL_PATH = "liuhaotian/llava-v1.6-mistral-7b"
TRAIN_DATASET = "HoqueMahmudul/clef2024_train_valid_a"
TEST_DATASET = "HoqueMahmudul/test_dataset"
LLAVA_REPO_URL = "https://github.com/haotian-liu/LLaVA.git"
LLAVA_REPO_DIR = "LLaVA"
OUTPUT_DIR = "Models/llava-v1.6-mistral-7b-r32-Mistral"
PREDICT_CSV = "Test dataset caption/llava-1.6-Mistral-r32-usual-prompt-matches more.csv"
EVALUATE_CSV = "captions_comparison.csv"

PROMPT = "What do you see in this picture?"
IGNORE_INDEX = -100

EPOCHS = 3
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 8
LEARNING_RATE = 1e-5
SAVE_EVAL_STEPS = 548
SAVE_TOTAL_LIMIT = 12
EARLY_STOPPING_PATIENCE = 3
EARLY_STOPPING_THRESHOLD = 0.01
MAX_NEW_TOKENS = 100

LORA_RANK = 32
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj",       # attention
    "fc1", "fc2",                       # vision tower MLP
    "mm_projector.0", "mm_projector.2",  # multimodal projector
    "up_proj", "down_proj", "gate_proj",  # language model MLP
]
FULLY_TRAINABLE_MODULES = ["lm_head", "embed_tokens"]


# --- Setup ---

def prepare_llava_repo(repo_dir=LLAVA_REPO_DIR):
    """Clone the LLaVA repository and patch llava/model/builder.py.

    Two patches are applied so the model loads cleanly in bfloat16:
      1. The vision tower is loaded (and cast to bfloat16) only if it is not already loaded,
         instead of being reloaded unconditionally.
      2. Every `float16` in builder.py is replaced with `bfloat16`.
    After this, install the repository once with `pip install -e LLaVA`.
    """
    if not os.path.isdir(repo_dir):
        subprocess.run(["git", "clone", LLAVA_REPO_URL, repo_dir], check=True)
    else:
        print("LLaVA directory already exists. Skipping clone.")

    builder_path = os.path.join(repo_dir, "llava", "model", "builder.py")
    with open(builder_path, "r") as f:
        content = f.read()

    pattern_block = (
        r"(vision_tower = model.get_vision_tower\(\)\n)"
        r".*?"
        r"(image_processor = vision_tower.image_processor)"
    )
    replacement_block = (
        r"\1"
        "        if not vision_tower.is_loaded:\n"
        "            print('vision_tower is not loaded so loading it now')\n"
        "            vision_tower.load_model(device_map=device_map)\n"
        "            vision_tower.to(device=device, dtype=torch.bfloat16)\n"
        "        else:\n"
        "            print('vision_tower is loaded')\n"
        r"        \2"
    )
    content = re.sub(pattern_block, replacement_block, content, flags=re.DOTALL)
    content = re.sub(r"(?<!b)float16", "bfloat16", content)

    with open(builder_path, "w") as f:
        f.write(content)
    print(f"Patched {builder_path}. Now run: pip install -e {repo_dir}")


# --- Prompting ---

def infer_conv_mode(model_name):
    """Pick the LLaVA conversation template that matches the checkpoint name."""
    name = model_name.lower()
    if "llama-2" in name:
        return "llava_llama_2"
    if "mistral" in name:
        return "mistral_instruct"
    if "v1.6-34b" in name:
        return "chatml_direct"
    if "v1" in name:
        return "llava_v1"
    if "mpt" in name:
        return "mpt"
    return "llava_v0"


def create_prompt(query, model, model_name, caption=None):
    """Build a conversation prompt with the image token; append the caption as the answer if given."""
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in query:
        if model.config.mm_use_im_start_end:
            query = re.sub(IMAGE_PLACEHOLDER, image_token_se, query)
        else:
            query = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, query)
    else:
        if model.config.mm_use_im_start_end:
            query = image_token_se + "\n" + query
        else:
            query = DEFAULT_IMAGE_TOKEN + "\n" + query

    conv = conv_templates[infer_conv_mode(model_name)].copy()
    conv.append_message(conv.roles[0], query)
    conv.append_message(conv.roles[1], caption)
    return conv.get_prompt()


# --- Images ---

def load_image(image_input):
    """Return a PIL image from a URL, a file path or an existing PIL image."""
    if isinstance(image_input, str):
        if image_input.startswith("http"):
            response = requests.get(image_input)
            return Image.open(BytesIO(response.content)).convert("RGB")
        return Image.open(image_input).convert("RGB")
    if isinstance(image_input, Image.Image):
        return image_input
    raise ValueError("Unsupported image input type")


def process_images(images, image_processor):
    """Run the image processor on each image and stack the results.

    Replaces llava.mm_utils.process_images: each image is processed on its own (no
    any-resolution tiling), so every sample becomes a single 336x336 view.
    """
    processed_images = []
    for image in images:
        pixel_values = image_processor(image)["pixel_values"]
        if isinstance(pixel_values, list):
            pixel_values = torch.tensor(pixel_values)
        processed_images.append(pixel_values)
    return torch.stack(processed_images, dim=0)


def process_and_prepare_images(image_files, image_processor, device):
    """Load and preprocess images; return a bfloat16 tensor and the original sizes."""
    images = [load_image(image_file) for image_file in image_files]
    image_sizes = [image.size for image in images]
    images_tensor = process_images(images, image_processor).to(device, dtype=torch.bfloat16)
    return images_tensor, image_sizes


# --- Model ---

def load_model(model_path, cache_dir, adapter_path=None):
    """Load the LLaVA model in bfloat16 and optionally attach a trained LoRA checkpoint."""
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path=model_path,
        model_base=None,
        model_name=model_name,
        cache_dir=cache_dir,
        use_flash_attn=True,
    )
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    return tokenizer, model, image_processor, model_name


def add_hybrid_adapters(model):
    """Wrap the model with LoRA adapters and fully unfreeze lm_head and embed_tokens."""
    config = LoraConfig(
        r=LORA_RANK,
        # Rescales the effective learning rate to LR * alpha / sqrt(r).
        lora_alpha=LORA_RANK * math.sqrt(LORA_RANK),
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(model, config)
    for name, param in model.named_parameters():
        if any(module in name for module in FULLY_TRAINABLE_MODULES):
            param.requires_grad_(True)
    model.print_trainable_parameters()
    return model


# --- Data ---

def tokenize_and_create_labels(example_batch, image_processor, tokenizer, model, device, model_name,
                               query=PROMPT, ignore_index=IGNORE_INDEX):
    """Build model inputs for a batch; only the caption tokens contribute to the loss."""
    pad_token_id = tokenizer.pad_token_id
    images_tensor, image_sizes = process_and_prepare_images(example_batch["image"], image_processor, device)

    prompt_only = [
        tokenizer_image_token(create_prompt(query, model, model_name, None), tokenizer,
                              IMAGE_TOKEN_INDEX, return_tensors="pt")
        for _ in example_batch["caption"]
    ]
    prompt_with_caption = [
        tokenizer_image_token(create_prompt(query, model, model_name, caption), tokenizer,
                              IMAGE_TOKEN_INDEX, return_tensors="pt")
        for caption in example_batch["caption"]
    ]

    input_ids = pad_sequence([ids.squeeze(0) for ids in prompt_with_caption],
                             batch_first=True, padding_value=pad_token_id).to(device)
    attention_mask = (input_ids != pad_token_id).long().to(device)

    labels = torch.full_like(input_ids, fill_value=ignore_index)
    for i, ids in enumerate(prompt_only):
        prompt_len = len(ids.squeeze(0))
        labels[i, prompt_len:] = input_ids[i, prompt_len:]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "images": images_tensor,
        "image_sizes": image_sizes,
        "labels": labels,
    }


# --- Training ---

class PrintBestCheckpointCallback(TrainerCallback):
    """Print the path of the best checkpoint when training ends."""

    def on_train_end(self, args, state, control, **kwargs):
        print(f"Best model checkpoint: {state.best_model_checkpoint}")


def train(args):
    """Fine-tune the model with LoRA + unfrozen lm_head/embed_tokens."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer, model, image_processor, model_name = load_model(args.model_path, args.cache_dir)

    def transform_batch(batch):
        return tokenize_and_create_labels(batch, image_processor, tokenizer, model, device, model_name)

    ds = load_dataset(args.dataset)
    train_ds = ds["train"]
    eval_ds = ds["test"]
    train_ds.set_transform(transform_batch)
    eval_ds.set_transform(transform_batch)

    model = add_hybrid_adapters(model)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        bf16=True,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        dataloader_pin_memory=False,
        save_total_limit=SAVE_TOTAL_LIMIT,
        evaluation_strategy="steps",
        save_strategy="steps",
        save_steps=SAVE_EVAL_STEPS,
        eval_steps=SAVE_EVAL_STEPS,
        logging_steps=1,
        num_train_epochs=args.epochs,
        remove_unused_columns=False,
        push_to_hub=False,
        label_names=["labels"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=None,
        optim="adamw_torch",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": True},
    )
    callbacks = [
        PrintBestCheckpointCallback(),
        EarlyStoppingCallback(
            early_stopping_patience=EARLY_STOPPING_PATIENCE,
            early_stopping_threshold=EARLY_STOPPING_THRESHOLD,
        ),
    ]
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        callbacks=callbacks,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)


# --- Inference ---

def generate_caption(tokenizer, model, image_processor, model_name, image, query,
                     temperature=1.0, num_beams=1, max_new_tokens=MAX_NEW_TOKENS):
    """Generate a caption for one image (greedy decoding by default)."""
    disable_torch_init()
    prompt = create_prompt(query, model, model_name)
    images = image if isinstance(image, list) else [image]
    images_tensor, image_sizes = process_and_prepare_images(images, image_processor, model.device)

    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .to(model.device)
    )
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images_tensor,
            image_sizes=image_sizes,
            do_sample=temperature != 1.0,
            temperature=temperature,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )
    return tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()


def predict(args):
    """Caption every image in the test set and write (ID, predicted_caption) to CSV."""
    tokenizer, model, image_processor, model_name = load_model(args.model_path, args.cache_dir, args.adapter_path)
    test_ds = load_dataset(args.test_dataset)["test"]

    captions = []
    for example in test_ds:
        predicted_caption = generate_caption(tokenizer, model, image_processor, model_name,
                                             example["image"], args.prompt,
                                             max_new_tokens=args.max_new_tokens)
        captions.append((example["image_name"], predicted_caption))

    df = pd.DataFrame(captions, columns=["ID", "predicted_caption"])
    print(df)
    write_csv(df, args.output_csv or PREDICT_CSV)


# --- Evaluation ---

def evaluate(args):
    """Caption the validation split and write (correct_caption, predicted_caption) to CSV."""
    tokenizer, model, image_processor, model_name = load_model(args.model_path, args.cache_dir, args.adapter_path)
    eval_ds = load_dataset(args.dataset)["test"]

    captions = []
    for example in eval_ds:
        predicted_caption = generate_caption(tokenizer, model, image_processor, model_name,
                                             example["image"], args.prompt,
                                             max_new_tokens=args.max_new_tokens)
        captions.append((example["caption"], predicted_caption))

    df = pd.DataFrame(captions, columns=["correct_caption", "predicted_caption"])
    print(df)
    write_csv(df, args.output_csv or EVALUATE_CSV)


def write_csv(df, path):
    """Write a DataFrame to CSV, creating the parent directory if needed."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved {len(df)} rows to {path}")


# --- Main ---

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=["setup", "train", "predict", "evaluate"], required=True)
    parser.add_argument("--model-path", default=MODEL_PATH, help="Base LLaVA checkpoint on the HF Hub.")
    parser.add_argument("--dataset", default=TRAIN_DATASET, help="HF dataset with train/test (validation) splits.")
    parser.add_argument("--test-dataset", default=TEST_DATASET, help="HF dataset with the unlabeled test split.")
    parser.add_argument("--llava-repo-dir", default=LLAVA_REPO_DIR)
    parser.add_argument("--cache-dir", default="", help="Model cache directory passed to LLaVA's loader.")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="Trainer checkpoint directory.")
    parser.add_argument("--resume-from-checkpoint", default=None, help="Checkpoint directory to resume training from.")
    parser.add_argument("--adapter-path", default=None, help="Trained LoRA checkpoint to load for predict/evaluate.")
    parser.add_argument("--output-csv", default=None,
                        help=f"CSV path (default: '{PREDICT_CSV}' for predict, '{EVALUATE_CSV}' for evaluate).")
    parser.add_argument("--epochs", type=float, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--prompt", default=PROMPT, help="User question placed after the image token.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.mode == "setup":
        prepare_llava_repo(args.llava_repo_dir)
        return
    if not llava_available:
        sys.exit("LLaVA is not installed. Run `--mode setup` and then `pip install -e LLaVA` first.")
    if args.mode == "train":
        train(args)
    elif args.mode == "predict":
        predict(args)
    elif args.mode == "evaluate":
        evaluate(args)


if __name__ == "__main__":
    main()
