"""Selective (LoRA) fine-tuning of LLaVA 1.6 Vicuna 7B for ImageCLEFmedical 2024 caption prediction.

Model: liuhaotian/llava-v1.6-vicuna-7b (Vicuna-7B language model + CLIP ViT-L/14-336
vision tower), loaded through the original LLaVA code base with FlashAttention 2.
LoRA adapters (r=16) are attached to the q/k/v attention projections, the vision-tower
MLP and the down/gate MLP projections of the language model; everything else stays
frozen. Data is the ROCOv2-based ImageCLEFmedical 2024 set on the
HuggingFace Hub.

Modes:
    setup     clone the LLaVA repository and patch its model builder (one time)
    train     fine-tune with the HuggingFace Trainer (early stopping on eval loss)
    preview   print predicted vs. reference captions for the first few validation images
    predict   caption every image of the test set and write an ID,predicted_caption CSV
    evaluate  caption the validation set and write a correct_caption,predicted_caption CSV
    plot      plot training / validation loss from a checkpoint's trainer_state.json

Usage:
    python llava_1_6_vicuna_7b_selective.py --mode setup
    pip install -e LLaVA
    python llava_1_6_vicuna_7b_selective.py --mode train
    python llava_1_6_vicuna_7b_selective.py --mode predict --adapter-path <checkpoint dir>

Environment variables:
    HF_TOKEN        HuggingFace token for the datasets (or run `huggingface-cli login`)
    WANDB_API_KEY   optional; the Trainer reports to every installed logger (wandb was used)

Requirements:
    transformers==4.37.2, accelerate==0.21.0, sentencepiece==0.1.99, torch (2.1.2 was used),
    torchvision, flash-attn, datasets, peft, bitsandbytes, loralib, hf_transfer, pandas, pillow,
    requests, matplotlib, and the LLaVA repo (https://github.com/haotian-liu/LLaVA)
    installed with `pip install -e LLaVA` after `--mode setup`.
"""

import argparse
import json
import os
import re
import subprocess
from io import BytesIO

# Faster model downloads; must be set before the HuggingFace libraries are imported.
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

import matplotlib.pyplot as plt
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
except ImportError as exc:  # lets `--mode setup` run before LLaVA is installed
    LLAVA_IMPORT_ERROR = exc
else:
    LLAVA_IMPORT_ERROR = None

# --- Defaults (values used in the original experiment) ---

LLAVA_REPO_URL = "https://github.com/haotian-liu/LLaVA.git"
LLAVA_REPO_DIR = "LLaVA"
MODEL_ID = "liuhaotian/llava-v1.6-vicuna-7b"
DATASET_ID = "HoqueMahmudul/clef2024_train_valid_a"
TEST_DATASET_ID = "HoqueMahmudul/test_dataset"
OUTPUT_DIR = "Models/llava-v1.6-vicuna-7b-Vicuna"
PREDICTIONS_CSV = "Test dataset caption/llava-1.6-vicuna-4384.csv"
EVAL_CSV = "captions_comparison.csv"
ADAPTER_PATH = os.path.join(OUTPUT_DIR, "checkpoint-4384")
TRAINER_STATE = os.path.join(OUTPUT_DIR, "checkpoint-4932", "trainer_state.json")

PROMPT = "What do you see in this picture?"
IGNORE_INDEX = -100

EPOCHS = 5
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 8
LEARNING_RATE = 1e-4
SAVE_EVAL_STEPS = 548
SAVE_TOTAL_LIMIT = 10
EARLY_STOPPING_PATIENCE = 3
EARLY_STOPPING_THRESHOLD = 0.01
MAX_NEW_TOKENS = 512
NUM_PREVIEW_IMAGES = 5

# The saved notebook cell for this config does not parse (a stray comment and a
# missing comma that fused "mm_projector.2" and "up_proj"). This is the target list
# that reproduces the 34,340,864 trainable parameters reported by the actual run,
# i.e. the projector and up_proj layers were not adapted.
LORA_CONFIG = dict(
    r=16,
    lora_alpha=32,
    target_modules=[
        "q_proj", "k_proj", "v_proj",  # attention
        "fc1", "fc2",  # vision tower MLP
        "down_proj", "gate_proj",  # language model MLP
    ],
    lora_dropout=0.05,
    bias="none",
)


# --- LLaVA setup ---

def prepare_llava_repo(repo_dir=LLAVA_REPO_DIR):
    """Clone the LLaVA repo (if missing) and patch llava/model/builder.py.

    The stock builder loads the vision tower in float16 and only when the model is
    not a LoRA checkpoint. The patch always loads the vision tower if needed and
    switches every float16 to bfloat16 so the whole model trains in bf16.
    """
    if not os.path.isdir(repo_dir):
        subprocess.run(["git", "clone", LLAVA_REPO_URL, repo_dir], check=True)
    else:
        print(f"{repo_dir} directory already exists. Skipping clone.")

    builder_path = os.path.join(repo_dir, "llava", "model", "builder.py")
    with open(builder_path, "r") as file:
        content = file.read()

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

    with open(builder_path, "w") as file:
        file.write(content)
    print(f"Patched {builder_path}. Now run: pip install -e {repo_dir}")


# --- Prompt ---

def infer_conv_mode(model_name):
    """Pick the LLaVA conversation template that matches the model name."""
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
    """Build the chat prompt with the image token; append the caption as the answer if given."""
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in query:
        if model.config.mm_use_im_start_end:
            query = re.sub(IMAGE_PLACEHOLDER, image_token_se, query)
        else:
            query = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, query)
    elif model.config.mm_use_im_start_end:
        query = image_token_se + "\n" + query
    else:
        query = DEFAULT_IMAGE_TOKEN + "\n" + query

    conv = conv_templates[infer_conv_mode(model_name)].copy()
    conv.append_message(conv.roles[0], query)
    conv.append_message(conv.roles[1], caption)
    return conv.get_prompt()


# --- Images ---

def load_image(image_input):
    """Return a PIL image from a URL, a file path, or an existing PIL image."""
    if isinstance(image_input, str):
        if image_input.startswith("http"):
            response = requests.get(image_input)
            return Image.open(BytesIO(response.content)).convert("RGB")
        return Image.open(image_input).convert("RGB")
    if isinstance(image_input, Image.Image):
        return image_input
    raise ValueError("Unsupported image input type")


def process_images(images, image_processor):
    """Run the CLIP image processor on each image and stack the pixel values.

    Replaces llava.mm_utils.process_images (no padding / any-res tiling).
    """
    processed_images = []
    for image in images:
        pixel_values = image_processor(image)["pixel_values"]
        if isinstance(pixel_values, list):
            pixel_values = torch.tensor(pixel_values)
        processed_images.append(pixel_values)
    return torch.stack(processed_images, dim=0)


def process_and_prepare_images(image_files, image_processor, device):
    """Load images and return a bf16 pixel tensor plus the original image sizes."""
    images = [load_image(image_file) for image_file in image_files]
    image_sizes = [image.size for image in images]
    images_tensor = process_images(images, image_processor).to(device, dtype=torch.bfloat16)
    return images_tensor, image_sizes


# --- Data ---

def tokenize_and_create_labels(example_batch, image_processor, tokenizer, model, device, model_name,
                               query=PROMPT, ignore_index=IGNORE_INDEX):
    """Turn a batch of (image, caption) rows into model inputs.

    Labels are the tokenised prompt+caption with every prompt token masked, so the
    loss is computed on the caption only.
    """
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


def load_train_eval_datasets(dataset_id, image_processor, tokenizer, model, device, model_name):
    """Load the train / validation splits with on-the-fly tokenisation."""
    def transform_batch(batch):
        return tokenize_and_create_labels(batch, image_processor, tokenizer, model, device, model_name)

    ds = load_dataset(dataset_id)
    train_ds, eval_ds = ds["train"], ds["test"]
    train_ds.set_transform(transform_batch)
    eval_ds.set_transform(transform_batch)
    return train_ds, eval_ds


# --- Model ---

def load_model(model_id, adapter_path=None):
    """Load the LLaVA base model, optionally with a trained LoRA adapter on top."""
    model_name = get_model_name_from_path(model_id)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_id,
        model_base=None,
        model_name=model_name,
        cache_dir="",
        use_flash_attn=True,
    )
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    return tokenizer, model, image_processor, model_name


def add_lora(model):
    """Wrap the model with the LoRA adapters; all other weights stay frozen."""
    model = get_peft_model(model, LoraConfig(**LORA_CONFIG))
    model.print_trainable_parameters()
    return model


# --- Training ---

class PrintBestCheckpointCallback(TrainerCallback):
    """Print the best checkpoint path when training ends."""

    def on_train_end(self, args, state, control, **kwargs):
        print(f"Best model checkpoint: {state.best_model_checkpoint}")


def train(args):
    """Fine-tune the LoRA adapters with the HuggingFace Trainer."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer, model, image_processor, model_name = load_model(args.model_id)
    train_ds, eval_ds = load_train_eval_datasets(args.dataset_id, image_processor, tokenizer,
                                                 model, device, model_name)
    model = add_lora(model)

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
    )
    callbacks = [
        PrintBestCheckpointCallback(),
        EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE,
                              early_stopping_threshold=EARLY_STOPPING_THRESHOLD),
    ]
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        callbacks=callbacks,
    )
    trainer.train()


def plot_losses(trainer_state_path, tick_step, output_file):
    """Plot training and validation loss from a trainer_state.json file."""
    with open(trainer_state_path, "r") as file:
        trainer_state = json.load(file)

    training_steps, training_losses = [], []
    validation_steps, validation_losses = [], []
    for log_entry in trainer_state["log_history"]:
        if "loss" in log_entry and "step" in log_entry:
            training_steps.append(log_entry["step"])
            training_losses.append(log_entry["loss"])
        elif "eval_loss" in log_entry and "step" in log_entry:
            validation_steps.append(log_entry["step"])
            validation_losses.append(log_entry["eval_loss"])

    fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    ax[0].plot(training_steps, training_losses, label="Training Loss", color="blue")
    ax[0].set_title("Training Loss")
    ax[0].set_ylabel("Loss")
    ax[0].grid(True)
    ax[1].plot(validation_steps, validation_losses, label="Validation Loss", color="orange")
    ax[1].set_title("Validation Loss")
    ax[1].set_ylabel("Loss")
    ax[1].set_xlabel("Steps")
    ax[1].grid(True)

    last_step = max(max(training_steps, default=0), max(validation_steps, default=0))
    ax[1].set_xticks(range(0, last_step + tick_step, tick_step))
    plt.tight_layout()
    fig.savefig(output_file)
    print(f"Saved loss plot to {output_file}")


# --- Inference ---

def generate_caption(tokenizer, model, image_processor, image, query, model_name,
                     max_new_tokens=MAX_NEW_TOKENS, temperature=1.0, num_beams=1):
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


def preview(args):
    """Print predicted and reference captions for the first few validation images."""
    tokenizer, model, image_processor, model_name = load_model(args.model_id, args.adapter_path)
    eval_ds = load_dataset(args.dataset_id)["test"]
    for i in range(min(NUM_PREVIEW_IMAGES, len(eval_ds))):
        predicted = generate_caption(tokenizer, model, image_processor, eval_ds[i]["image"],
                                     args.prompt, model_name, args.max_new_tokens)
        print(f"Predicted caption: {predicted}")
        print(f"Correct caption: {eval_ds[i]['caption']}\n")


def predict(args):
    """Caption every test image and save an ID,predicted_caption CSV."""
    tokenizer, model, image_processor, model_name = load_model(args.model_id, args.adapter_path)
    test_ds = load_dataset(args.test_dataset_id)["test"]
    captions = []
    for i in range(len(test_ds)):
        predicted = generate_caption(tokenizer, model, image_processor, test_ds[i]["image"],
                                     args.prompt, model_name, args.max_new_tokens)
        captions.append((test_ds[i]["image_name"], predicted))

    df = pd.DataFrame(captions, columns=["ID", "predicted_caption"])
    print(df)
    os.makedirs(os.path.dirname(args.predictions_csv) or ".", exist_ok=True)
    df.to_csv(args.predictions_csv, index=False)


# --- Evaluation ---

def evaluate(args):
    """Caption the validation split and save reference vs. predicted captions to CSV."""
    tokenizer, model, image_processor, model_name = load_model(args.model_id, args.adapter_path)
    eval_ds = load_dataset(args.dataset_id)["test"]
    captions = []
    for i in range(len(eval_ds)):
        predicted = generate_caption(tokenizer, model, image_processor, eval_ds[i]["image"],
                                     args.prompt, model_name, args.max_new_tokens)
        captions.append((eval_ds[i]["caption"], predicted))

    df = pd.DataFrame(captions, columns=["correct_caption", "predicted_caption"])
    print(df)
    df.to_csv(args.eval_csv, index=False)


# --- Main ---

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", required=True,
                        choices=["setup", "train", "preview", "predict", "evaluate", "plot"])
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--dataset-id", default=DATASET_ID, help="train/validation dataset")
    parser.add_argument("--test-dataset-id", default=TEST_DATASET_ID)
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="where checkpoints are saved")
    parser.add_argument("--adapter-path", default=ADAPTER_PATH,
                        help="LoRA checkpoint to load for preview/predict/evaluate (omit for the base model)")
    parser.add_argument("--epochs", type=float, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--predictions-csv", default=PREDICTIONS_CSV)
    parser.add_argument("--eval-csv", default=EVAL_CSV)
    parser.add_argument("--trainer-state", default=TRAINER_STATE, help="trainer_state.json for --mode plot")
    parser.add_argument("--plot-file", default="loss_curves.png")
    parser.add_argument("--llava-repo-dir", default=LLAVA_REPO_DIR)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.mode == "setup":
        prepare_llava_repo(args.llava_repo_dir)
        return
    if args.mode == "plot":
        plot_losses(args.trainer_state, SAVE_EVAL_STEPS, args.plot_file)
        return
    if LLAVA_IMPORT_ERROR is not None:
        raise SystemExit(f"LLaVA is not importable ({LLAVA_IMPORT_ERROR}). "
                         "Run --mode setup, then `pip install -e LLaVA`.")

    if args.mode == "train":
        train(args)
    elif args.mode == "preview":
        preview(args)
    elif args.mode == "predict":
        predict(args)
    elif args.mode == "evaluate":
        evaluate(args)


if __name__ == "__main__":
    main()
