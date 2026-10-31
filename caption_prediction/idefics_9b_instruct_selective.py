"""Selective fine-tuning of IDEFICS 9B Instruct for ImageCLEFmedical 2024 caption prediction.

Model: HuggingFaceM4/idefics-9b-instruct (Flamingo-style model: CLIP ViT-H vision encoder,
perceiver resampler and a LLaMA-7B decoder with gated cross-attention), loaded in 4-bit
NF4 with double quantization and bfloat16 compute. `lm_head` and `embed_tokens` are kept
out of quantization.

"Selective" fine-tuning: only LoRA adapters (r=16, alpha=32) on the attention projections
(q_proj, k_proj, v_proj, out_proj) of the vision encoder, perceiver and the decoder's
self-/cross-attention are trained (21,061,632 trainable parameters, 0.24%).

The script can:
  * --mode train         fine-tune on the ROCOv2 train/validation split (HF dataset)
  * --mode predict       caption the unlabeled test set and write a CSV (Image Name, Caption)
  * --mode plot-history  plot training/validation loss from a checkpoint's trainer_state.json

Usage:
    python idefics_9b_instruct_selective.py --mode train
    python idefics_9b_instruct_selective.py --mode train --resume-from-checkpoint "Models/<run>/checkpoint-6500"
    python idefics_9b_instruct_selective.py --mode predict --adapter-path "Models/<run>/checkpoint-XXXX"
    python idefics_9b_instruct_selective.py --mode plot-history --trainer-state "Models/<run>/checkpoint-XXXX/trainer_state.json"

Environment:
    HF_TOKEN       Hugging Face token for the private datasets (or run `huggingface-cli login`).
    WANDB_API_KEY  Optional. `report_to=None` reports to every installed integration, so
                   Weights & Biases is used when it is installed and logged in.

Requirements (the original run used the then-latest releases; transformers must still
accept `evaluation_strategy`, i.e. < 4.46):
    torch (2.3.0 was used), transformers, accelerate, bitsandbytes, sentencepiece, loralib,
    peft (installed from git), datasets, hf_transfer, pandas, matplotlib, Pillow
"""

import argparse
import json
import os
import re

# Must be set before huggingface_hub is imported for faster downloads.
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

import matplotlib.pyplot as plt
import pandas as pd
import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    IdeficsForVisionText2Text,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)


# --- Defaults ---

CHECKPOINT = "HuggingFaceM4/idefics-9b-instruct"
TRAIN_DATASET = "HoqueMahmudul/clef2024_train_valid_a"
TEST_DATASET = "HoqueMahmudul/test_dataset"
OUTPUT_DIR = "Models/idefics-9b-instruct-2cuda-r16-params 21,061,632"
PREDICT_CSV = "Test dataset caption/idefics_9B_r16-params 21,061,632.csv"

USER_PROMPT = "User: What do you see in this picture?"
ASSISTANT_PREFIX = "\nAssistant: "

EPOCHS = 2
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 8
LEARNING_RATE = 1e-4
SAVE_EVAL_STEPS = 500
SAVE_TOTAL_LIMIT = 25
LOGGING_STEPS = 10
EARLY_STOPPING_PATIENCE = 1
EARLY_STOPPING_THRESHOLD = 0.5
MAX_NEW_TOKENS = 50
PLOT_TICK_STEP = 1095

LORA_CONFIG = dict(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
    lora_dropout=0.05,
    bias="none",
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# --- Model ---

def load_model(checkpoint, cache_dir, adapter_path=None):
    """Load the processor and the 4-bit quantized model, optionally with a trained LoRA checkpoint."""
    # lm_head and embed_tokens are skipped because they cannot be quantized properly.
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        llm_int8_skip_modules=["lm_head", "embed_tokens"],
    )
    processor = AutoProcessor.from_pretrained(checkpoint)
    model = IdeficsForVisionText2Text.from_pretrained(
        checkpoint,
        quantization_config=bnb_config,
        device_map="auto",
        cache_dir=cache_dir,
    )
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    return processor, model


def add_lora_adapters(model):
    """Wrap the model with LoRA adapters on the attention projections."""
    model = get_peft_model(model, LoraConfig(**LORA_CONFIG))
    model.print_trainable_parameters()
    return model


# --- Data ---

def make_transform(processor):
    """Return a dataset transform that turns (image, caption) pairs into model inputs."""

    def ds_transforms(example_batch):
        prompts = [
            [image, USER_PROMPT, f"{ASSISTANT_PREFIX}{caption}"]
            for image, caption in zip(example_batch["image"], example_batch["caption"])
        ]
        inputs = processor(prompts, return_tensors="pt").to(DEVICE)
        # Labels are the full input ids: the loss covers the prompt as well as the caption.
        inputs["labels"] = inputs["input_ids"]
        return inputs

    return ds_transforms


# --- Training ---

class PrintBestCheckpointCallback(TrainerCallback):
    """Print the path of the best checkpoint when training ends."""

    def on_train_end(self, args, state, control, **kwargs):
        print(f"Best model checkpoint: {state.best_model_checkpoint}")


def train(args):
    """Fine-tune the LoRA adapters with the Hugging Face Trainer."""
    processor, model = load_model(args.model_id, args.cache_dir)

    ds = load_dataset(args.dataset)
    train_ds = ds["train"]
    eval_ds = ds["test"]
    transform = make_transform(processor)
    train_ds.set_transform(transform)
    eval_ds.set_transform(transform)

    model = add_lora_adapters(model)

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
        logging_steps=LOGGING_STEPS,
        num_train_epochs=args.epochs,
        remove_unused_columns=False,
        push_to_hub=False,
        label_names=["labels"],
        load_best_model_at_end=True,
        report_to=None,
        optim="adamw_torch",
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

def check_inference(model, processor, image, prompt_text, max_new_tokens=MAX_NEW_TOKENS):
    """Generate text for one image with greedy decoding and strip special tokens."""
    tokenizer = processor.tokenizer
    bad_words_ids = tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids
    exit_condition = tokenizer("<end_of_utterance>", add_special_tokens=False).input_ids

    inputs = processor([image] + prompt_text, return_tensors="pt").to(DEVICE)
    generated_ids = model.generate(
        **inputs,
        eos_token_id=exit_condition,
        bad_words_ids=bad_words_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    return re.sub(r"<.*?>", "", generated_text).strip()


def clean_caption(caption):
    """Remove the echoed prompt so only the generated caption remains."""
    cleaned = re.sub(r"User: What do you see in this picture\?\s*Assistant: ", "", caption, flags=re.IGNORECASE)
    return cleaned.strip()


def predict(args):
    """Caption every image in the test set and write (Image Name, Caption) to CSV."""
    processor, model = load_model(args.model_id, args.cache_dir, args.adapter_path)
    test_ds = load_dataset(args.test_dataset)["test"]

    results = []
    for example in test_ds:
        generated = check_inference(model, processor, example["image"],
                                    [USER_PROMPT, ASSISTANT_PREFIX], args.max_new_tokens)
        results.append({"Image Name": example["image_name"], "Caption": clean_caption(generated)})

    df = pd.DataFrame(results)
    parent = os.path.dirname(args.output_csv)
    if parent:
        os.makedirs(parent, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    print(f"Saved {len(df)} rows to {args.output_csv}")


# --- Evaluation ---

def plot_history(trainer_state_path, tick_step=PLOT_TICK_STEP):
    """Plot training and validation loss from a Trainer `trainer_state.json`."""
    with open(trainer_state_path, "r") as f:
        trainer_state = json.load(f)

    training_steps, training_losses = [], []
    validation_steps, validation_losses = [], []
    for entry in trainer_state["log_history"]:
        if "loss" in entry and "step" in entry:
            training_steps.append(entry["step"])
            training_losses.append(entry["loss"])
        elif "eval_loss" in entry and "step" in entry:
            validation_steps.append(entry["step"])
            validation_losses.append(entry["eval_loss"])

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
    plt.show()


# --- Main ---

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=["train", "predict", "plot-history"], required=True)
    parser.add_argument("--model-id", default=CHECKPOINT, help="Base IDEFICS checkpoint on the HF Hub.")
    parser.add_argument("--dataset", default=TRAIN_DATASET, help="HF dataset with train/test (validation) splits.")
    parser.add_argument("--test-dataset", default=TEST_DATASET, help="HF dataset with the unlabeled test split.")
    parser.add_argument("--cache-dir", default="", help="Model cache directory passed to from_pretrained.")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="Trainer checkpoint directory.")
    parser.add_argument("--resume-from-checkpoint", default=None, help="Checkpoint directory to resume training from.")
    parser.add_argument("--adapter-path", default=None, help="Trained LoRA checkpoint to load for predict.")
    parser.add_argument("--output-csv", default=PREDICT_CSV)
    parser.add_argument("--trainer-state", default=None, help="trainer_state.json to plot (plot-history mode).")
    parser.add_argument("--epochs", type=float, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.mode == "train":
        train(args)
    elif args.mode == "predict":
        predict(args)
    elif args.mode == "plot-history":
        if not args.trainer_state:
            raise SystemExit("--trainer-state is required for plot-history")
        plot_history(args.trainer_state)


if __name__ == "__main__":
    main()
