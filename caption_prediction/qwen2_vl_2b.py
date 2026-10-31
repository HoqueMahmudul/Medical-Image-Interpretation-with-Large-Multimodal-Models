"""Caption prediction with Qwen2-VL-2B-Instruct (ImageCLEFmedical Caption 2025).

Qwen2-VL-2B-Instruct is a 2B-parameter vision-language chat model. This script
LoRA fine-tunes it on medical image/caption pairs and uses the adapter to
caption the ImageCLEFmedical 2025 validation and test images.

Modes:
* ``train``          - LoRA fine-tuning with the Hugging Face Trainer; the loss
                       is computed on the assistant (caption) tokens only.
                       Evaluation/checkpointing every 500 steps, early stopping.
* ``evaluate``       - caption the dataset's test split, report ANLS
                       (rapidfuzz ratio) against the reference captions.
* ``predict-valid``  - caption a folder of validation images listed in a
                       captions CSV (ID, Caption) and save ground truth vs.
                       generated captions.
* ``predict``        - caption every test image in a folder (checkpointing the
                       CSV every 1000 images), then strip leading " .:" chars.
* ``fill-missing``   - re-caption rows whose caption came out empty after
                       ``predict`` (images letterboxed to 512x512), then write
                       the completed CSV and an ID-sorted copy.
* ``predict-typed``  - final submission run: caption each test image and
                       prefix the image type predicted by a separate
                       classifier (CSV with ID, predicted_label).
* ``plot``           - plot training/validation loss from a trainer_state.json.

Usage:
    python qwen2_vl_2b.py --mode train --output-dir fine-tuned-model
    python qwen2_vl_2b.py --mode evaluate --adapter-dir fine-tuned-model/checkpoint-2500
    python qwen2_vl_2b.py --mode predict --adapter-dir ... --image-dir Test/test
    python qwen2_vl_2b.py --mode fill-missing --adapter-dir ... --image-dir Test/test
    python qwen2_vl_2b.py --mode predict-typed --adapter-dir ... --image-dir Test/test \
        --classification-csv test_predictions_Final.csv
    python qwen2_vl_2b.py --mode plot --trainer-state fine-tuned-model/checkpoint-2500/trainer_state.json

Environment:
    HF_TOKEN   Hugging Face token for gated/private models and datasets
               (or run ``huggingface-cli login`` once).
    Training logs go to every installed reporting integration (e.g. wandb, in
    which case set WANDB_API_KEY or run ``wandb login``).

Requirements:
    torch, torchvision, transformers, qwen-vl-utils, accelerate, datasets, peft,
    bitsandbytes, hf_transfer, flash_attn, tensorboard, pandas, pillow, tqdm,
    rapidfuzz, matplotlib
"""

import argparse
import json
import os
import re

# Faster model downloads; must be set before the HuggingFace libraries are imported.
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

import pandas as pd
import torch
from datasets import load_dataset
from PIL import Image
from peft import LoraConfig, PeftModel, get_peft_model
from rapidfuzz import fuzz
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    EarlyStoppingCallback,
    Qwen2VLForConditionalGeneration,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

# --- Defaults ---

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
DATASET_ID = "racho1/Clef_2025"
OUTPUT_DIR = "fine-tuned-model"
DEVICE = "cuda"
MODEL_MAX_LENGTH = 16290  # lower than the 32768 default to limit VRAM use

TRAIN_QUESTION = "What do you see here?"
PREDICT_VALID_QUESTION = "What do you see in this image?"
FILL_MISSING_PROMPT = "Describe all anatomical structures and abnormalities in this medical image:"
TYPED_PROMPT = (
    "Describe the anatomical structures and abnormalities in this medical image. "
    "Do not predict or guess the image type."
)

EPOCHS = 3
BATCH_SIZE = 1
LEARNING_RATE = 3e-4
EVAL_SAVE_STEPS = 500
MAX_NEW_TOKENS = 64
LONG_MAX_NEW_TOKENS = 200  # fill-missing and predict-typed
TARGET_SIZE = 512
SAVE_EVERY = 1000

# Words removed from raw captions in predict-typed so that only the classifier's
# image type names the modality.
FORBIDDEN_WORDS = [
    "ct", "ct scan", "computed tomography", "mri", "mri scan", "magnetic resonance",
    "radiograph", "radiography", "xray", "x-ray", "x ray",
    "ultrasound", "sonograph", "scan", "image", "chest", "plain", "postoperative",
]


# --- Model ---

def load_base_model(model_id, use_flash_attention=False, device_map=DEVICE):
    """Load Qwen2-VL in bfloat16."""
    kwargs = {}
    if use_flash_attention:
        kwargs["attn_implementation"] = "flash_attention_2" if DEVICE == "cuda" else "eager"
    return Qwen2VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map=device_map, **kwargs
    )


def load_finetuned(model_id, adapter_dir, device_map=DEVICE):
    """Load the base model plus a LoRA adapter checkpoint, with tokenizer and processor."""
    model = load_base_model(model_id, device_map=device_map)
    model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    processor = AutoProcessor.from_pretrained(model_id)
    return model, tokenizer, processor


def apply_lora(model):
    """Add LoRA adapters to the language model and the vision merger MLP."""
    lora_config = LoraConfig(
        r=32,
        lora_alpha=32,
        use_rslora=True,
        # "all-linear" causes issues with Qwen; the vision "fc1", "fc2", "qkv"
        # layers were left out because they easily run out of memory.
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj",
            "mlp.0", "mlp.2",
        ],
        modules_to_save=["lm_head", "embed_tokens"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


# --- Data ---

class AssistantOnlyCollator:
    """Build chat-formatted batches and mask every token except the caption (+ EOS)."""

    def __init__(self, processor):
        self.processor = processor

    def __call__(self, examples):
        texts, images, answers = [], [], []
        for example in examples:
            answer = example["caption"]
            messages = [
                {"role": "user", "content": [
                    {"type": "text", "text": TRAIN_QUESTION},
                    {"type": "image"},
                ]},
                {"role": "assistant", "content": [{"type": "text", "text": answer}]},
            ]
            text = self.processor.apply_chat_template(messages, add_generation_prompt=False)
            texts.append(text.strip())
            images.append([example["image"]])
            answers.append(answer)

        batch = self.processor(text=texts, images=images, return_tensors="pt", padding=True)
        labels = batch["input_ids"].clone()

        for i, (input_ids, answer) in enumerate(zip(batch["input_ids"], answers)):
            answer_tokens = self.processor.tokenizer(answer, return_tensors="pt")["input_ids"][0]
            start_idx = self.find_subsequence(input_ids, answer_tokens)
            if start_idx is not None:
                end_idx = start_idx + len(answer_tokens)
                # Keep the EOS token unmasked so the model learns to stop.
                if input_ids[end_idx] == self.processor.tokenizer.eos_token_id:
                    end_idx += 1
                labels[i, :start_idx] = -100
                labels[i, end_idx:] = -100

        batch["labels"] = labels
        return batch

    @staticmethod
    def find_subsequence(sequence, subsequence):
        """Return the start index of subsequence in sequence, or None."""
        seq_len, sub_len = len(sequence), len(subsequence)
        for i in range(seq_len - sub_len + 1):
            if torch.equal(sequence[i:i + sub_len], subsequence):
                return i
        return None


def letterbox(image, target_size=TARGET_SIZE):
    """Resize keeping aspect ratio and pad to a black target_size square."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    width, height = image.size
    scale = min(target_size / width, target_size / height)
    new_w, new_h = int(width * scale), int(height * scale)
    image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    result = Image.new("RGB", (target_size, target_size), (0, 0, 0))
    result.paste(image, ((target_size - new_w) // 2, (target_size - new_h) // 2))
    return result


def medical_resize(image_path, target_size=TARGET_SIZE):
    """Open an image file and letterbox it; return None if it cannot be read."""
    try:
        return letterbox(Image.open(image_path), target_size)
    except Exception as e:
        print(f"Resize failed for {image_path}: {e}")
        return None


def resolve_image_path(image_id, directory):
    """Find the image file whose name starts with image_id (case-insensitive)."""
    image_id_lower = image_id.lower()
    for fname in os.listdir(directory):
        if fname.lower().startswith(image_id_lower) and fname.lower().endswith((".jpg", ".jpeg", ".png")):
            return os.path.join(directory, fname)
    return None


# --- Inference ---

def generate_caption(model, processor, image, prompt, max_new_tokens=MAX_NEW_TOKENS, **gen_kwargs):
    """Caption one image with the processor-built chat prompt (text before image)."""
    messages = [{"role": "user", "content": [
        {"type": "text", "text": prompt},
        {"type": "image"},
    ]}]
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=[text.strip()], images=[image], return_tensors="pt", padding=True).to(DEVICE)
    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, **gen_kwargs)
    return processor.batch_decode(
        generated_ids[:, inputs["input_ids"].size(1):], skip_special_tokens=True
    )[0]


def generate_caption_tokenizer_only(model, tokenizer, image, prompt, max_new_tokens, **gen_kwargs):
    """Generate with the tokenizer's chat template, as the fill-missing/typed runs did.

    NOTE: the tokenizer cannot encode images, so no pixel values reach the
    model here; the caption is conditioned on the text prompt only. Kept as-is
    to reproduce the submitted runs.
    """
    messages = [{"role": "user", "content": [
        {"type": "text", "text": prompt},
        {"type": "image", "image": image},
    ]}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, **gen_kwargs)
    return tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()


def remove_invalid_start_chars(caption):
    """Strip leading spaces, periods and colons from a generated caption."""
    if isinstance(caption, str):
        return caption.lstrip(" .:")
    return caption


def clean_caption(generated, image_type):
    """Remove modality words from a caption and prefix the classifier's image type."""
    text = generated.lower()
    for word in FORBIDDEN_WORDS:
        text = re.sub(rf"\b{re.escape(word)}\b", "", text)
    text = re.sub(r"\b(shows|showed|demonstrated|displayed|revealed|indicates)\b", "", text)
    text = re.sub(r"[^a-zA-Z0-9., ]", "", text)
    text = " ".join(text.split()).strip(" ,.")
    if not text:
        return f"{image_type} shows no abnormality detected."
    return f"{image_type} shows {text[0].upper() + text[1:]}"


def predict(args, model, processor):
    """Caption every image in --image-dir, saving progress every SAVE_EVERY images."""
    os.makedirs(args.output_dir, exist_ok=True)
    test_image_ids = [os.path.splitext(f)[0] for f in os.listdir(args.image_dir)]
    image_ids, generated_texts = [], []
    batch_counter = 0

    for i, image_id in enumerate(tqdm(test_image_ids)):
        image_path = os.path.join(args.image_dir, f"{image_id}.jpg")
        if not os.path.exists(image_path):
            print(f"Warning: {image_path} does not exist. Skipping this image.")
            continue
        image_ids.append(image_id)
        generated_texts.append(
            generate_caption(model, processor, Image.open(image_path), TRAIN_QUESTION, args.max_new_tokens)
        )

        if (i + 1) % SAVE_EVERY == 0:
            pd.DataFrame({"ID": image_ids, "Generated_Caption": generated_texts}).to_csv(
                os.path.join(args.output_dir, f"generated_captions_test_set_batch_{batch_counter}.csv"),
                index=False,
            )
            print(f"Batch {batch_counter} saved to CSV.")
            batch_counter += 1

    output_df = pd.DataFrame({"ID": image_ids, "Generated_Caption": generated_texts})
    output_df.to_csv(os.path.join(args.output_dir, "generated_captions_test_set_final.csv"), index=False)
    print("Final batch saved to CSV.")

    output_df["Generated_Caption"] = output_df["Generated_Caption"].apply(remove_invalid_start_chars)
    cleaned_path = os.path.join(args.output_dir, "updated_generated_captions_test_set.csv")
    output_df.to_csv(cleaned_path, index=False)
    print(output_df.head())
    print(f"Cleaned captions saved to {cleaned_path}")


def predict_valid(args, model, processor):
    """Caption validation images listed in --captions-csv and keep the ground truth."""
    valid_caption_df = pd.read_csv(args.captions_csv)
    valid_caption_df["Ground_Truth_Caption"] = valid_caption_df["Caption"].str.strip()

    test_image_ids = [os.path.splitext(f)[0] for f in os.listdir(args.image_dir)]
    filtered = valid_caption_df[valid_caption_df["ID"].isin(test_image_ids)]

    ground_truths, generated_texts, image_ids = [], [], []
    for _, row in tqdm(filtered.iterrows(), total=len(filtered)):
        image = Image.open(os.path.join(args.image_dir, f"{row['ID']}.jpg"))
        caption = generate_caption(
            model, processor, image, PREDICT_VALID_QUESTION,
            max_new_tokens=args.max_new_tokens, temperature=0.7,
        )
        if caption.startswith("."):
            caption = caption[1:].strip()
        generated_texts.append(caption)
        ground_truths.append(row["Ground_Truth_Caption"])
        image_ids.append(row["ID"])

    output_df = pd.DataFrame({
        "ID": image_ids,
        "Ground_Truth_Caption": ground_truths,
        "Generated_Caption": generated_texts,
    })
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "new_captions_test_set.csv")
    output_df.to_csv(out_path, index=False)
    print(output_df.head())
    print(f"Saved to {out_path}")


def fill_missing(args, model, tokenizer):
    """Re-caption rows with an empty caption and save the completed CSV."""
    input_csv = args.input_csv or os.path.join(args.output_dir, "updated_generated_captions_test_set.csv")
    df = pd.read_csv(input_csv)
    missing = df[df["Generated_Caption"].isnull() | (df["Generated_Caption"] == "")].copy()
    print(f"Found {len(missing)} images needing captions")

    results = []
    for _, row in tqdm(missing.iterrows(), total=len(missing), desc="Processing"):
        image_path = f"{args.image_dir}/{row['ID']}.jpg"
        record = {"ID": row["ID"], "Status": "Pending"}

        if not os.path.exists(image_path):
            record.update({"Status": "Image not found", "Caption": ""})
            results.append(record)
            continue

        try:
            resized_img = medical_resize(image_path)
            if not resized_img:
                record.update({"Status": "Resize failed", "Caption": ""})
                results.append(record)
                continue

            caption = generate_caption_tokenizer_only(
                model, tokenizer, resized_img, FILL_MISSING_PROMPT,
                max_new_tokens=args.max_new_tokens,
                do_sample=True, temperature=0.7, top_p=0.9, num_beams=3,
            )
            with Image.open(image_path) as original:
                original_size = f"{original.size[0]}x{original.size[1]}"
            record.update({
                "Status": "Success",
                "Caption": caption,
                "Original_Size": original_size,
                "Resized_Size": f"{TARGET_SIZE}x{TARGET_SIZE}",
            })
            df.loc[df["ID"] == row["ID"], "Generated_Caption"] = caption
        except Exception as e:
            record.update({"Status": f"Error: {e}", "Caption": ""})

        results.append(record)

    os.makedirs(args.output_dir, exist_ok=True)
    new_captions_path = os.path.join(args.output_dir, "newly_generated_captions.csv")
    complete_path = os.path.join(args.output_dir, "updated_generated_captions_test_set_complete.csv")
    sorted_path = os.path.join(args.output_dir, "sorted_file.csv")
    pd.DataFrame(results).to_csv(new_captions_path, index=False)
    df.to_csv(complete_path, index=False)
    df.sort_values(by="ID").reset_index(drop=True).to_csv(sorted_path, index=False)

    print("\nCompleted processing. Results saved to:")
    print(f"- New captions: {new_captions_path}")
    print(f"- Updated complete dataset: {complete_path}")
    print(f"- Sorted by ID: {sorted_path}")


def predict_typed(args, model, tokenizer):
    """Caption each classified test image and inject the predicted image type."""
    df = pd.read_csv(args.classification_csv)
    results = []
    print(f"\nGenerating captions for {len(df)} images...\n")

    for _, row in tqdm(df.iterrows(), total=len(df)):
        img_id = str(row["ID"]).strip()
        img_type = str(row["predicted_label"]).strip()
        image_path = resolve_image_path(img_id, args.image_dir)
        record = {"ID": img_id, "Type": img_type}

        if not image_path:
            record["Caption"] = "Image not found"
            print(f"{img_id} -> Image not found")
            results.append(record)
            continue

        try:
            img = medical_resize(image_path)
            if img is None:
                record["Caption"] = "Resize failed"
                print(f"{img_id} -> Resize failed")
                results.append(record)
                continue
            raw = generate_caption_tokenizer_only(
                model, tokenizer, img, TYPED_PROMPT,
                max_new_tokens=args.max_new_tokens,
                do_sample=True, temperature=1.0, top_p=0.95, top_k=50, num_beams=3,
            )
            final = clean_caption(raw, img_type)
        except Exception as e:
            final = f"Error: {e}"

        print(f"{img_id} -> {final}")
        record["Caption"] = final
        results.append(record)

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "final_forced_type_captions.csv")
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"\nFinal captions saved to: {out_path}")


# --- Evaluation ---

def anls_score(predictions, references):
    """Average normalised Levenshtein similarity (rapidfuzz ratio), skipping empty predictions."""
    valid_preds, valid_refs = [], []
    for pred, ref in zip(predictions, references):
        pred = pred.strip().strip(".")
        ref = ref.strip().strip(".")
        if not pred:
            print("Warning: Skipped an empty prediction.")
            continue
        valid_preds.append(pred)
        valid_refs.append(ref)
    scores = [fuzz.ratio(p, r) / 100.0 for p, r in zip(valid_preds, valid_refs)]
    return (sum(scores) / len(scores) if scores else 0.0), len(valid_preds)


def evaluate(args, model, processor):
    """Caption the dataset's test split and report ANLS against the references."""
    eval_dataset = load_dataset(args.dataset_id)["test"]
    references, predictions = [], []
    for example in tqdm(eval_dataset):
        references.append(example["caption"])
        caption = generate_caption(
            model, processor, [example["image"]], TRAIN_QUESTION, args.max_new_tokens
        )
        predictions.append(caption)
        print(caption)

    predictions = [p.strip().strip(".") for p in predictions]
    anls, n_valid = anls_score(predictions, references)
    print(f"ANLS: {anls}")
    print(f"Total predictions: {len(predictions)}")
    print(f"Valid predictions used: {n_valid}")
    print(f"Skipped (empty): {len(predictions) - n_valid}")

    for i in range(min(10, len(predictions))):
        print(f"Example {i + 1}")
        print(f"Ground Truth : {references[i].strip()}")
        print(f"Prediction   : {predictions[i].strip()}")
        print("-" * 60)


# --- Training ---

class PrintBestCheckpointCallback(TrainerCallback):
    """Print the best checkpoint so far after every evaluation."""

    def on_evaluate(self, args, state, control, logs=None, **kwargs):
        if state.best_model_checkpoint:
            print(f"Best model checkpoint: {state.best_model_checkpoint}")
        return control


def train(args):
    """LoRA fine-tune Qwen2-VL on the caption dataset."""
    model = load_base_model(args.model_id, use_flash_attention=True)
    processor = AutoProcessor.from_pretrained(args.model_id)
    processor.tokenizer.model_max_length = MODEL_MAX_LENGTH

    ds = load_dataset(args.dataset_id)
    train_dataset, eval_dataset = ds["train"], ds["test"]

    model = apply_lora(model)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        bf16=True,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        dataloader_pin_memory=False,
        save_total_limit=1,
        eval_strategy="steps",
        save_strategy="steps",
        save_steps=EVAL_SAVE_STEPS,
        eval_steps=EVAL_SAVE_STEPS,
        logging_steps=1,
        weight_decay=0.01,
        num_train_epochs=args.epochs,
        remove_unused_columns=False,
        push_to_hub=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=None,  # None = all installed integrations
        gradient_checkpointing=True,
        # Reentrant checkpointing leaves Qwen's LoRA weights without gradients.
        gradient_checkpointing_kwargs={"use_reentrant": False},
        resume_from_checkpoint=True,  # informational only; see --resume
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=AssistantOnlyCollator(processor),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[
            PrintBestCheckpointCallback(),
            EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.01),
        ],
        tokenizer=processor,
    )
    trainer.train(resume_from_checkpoint=True if args.resume else None)


def plot_losses(trainer_state_path, out_path):
    """Plot training and validation loss curves from a trainer_state.json file."""
    import matplotlib.pyplot as plt

    with open(trainer_state_path) as f:
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

    tick_step = 500
    max_step = max(max(training_steps, default=0), max(validation_steps, default=0))
    ax[1].set_xticks(range(0, max_step + tick_step, tick_step))
    plt.tight_layout()
    fig.savefig(out_path)
    print(f"Loss plot saved to {out_path}")


# --- CLI ---

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--mode", required=True, choices=[
        "train", "evaluate", "predict-valid", "predict", "fill-missing", "predict-typed", "plot",
    ])
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--dataset-id", default=DATASET_ID)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--adapter-dir", default=None,
                        help="LoRA checkpoint to load for the inference modes")
    parser.add_argument("--image-dir", default=None, help="folder of .jpg images to caption")
    parser.add_argument("--captions-csv", default="valid_captions.csv",
                        help="validation CSV with ID and Caption columns (predict-valid)")
    parser.add_argument("--input-csv", default=None,
                        help="CSV with empty captions to fill (fill-missing); "
                             "default: <output-dir>/updated_generated_captions_test_set.csv")
    parser.add_argument("--classification-csv", default="test_predictions_Final.csv",
                        help="CSV with ID and predicted_label columns (predict-typed)")
    parser.add_argument("--trainer-state", default=None, help="trainer_state.json to plot (plot)")
    parser.add_argument("--epochs", type=float, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--max-new-tokens", type=int, default=None,
                        help=f"default: {LONG_MAX_NEW_TOKENS} for fill-missing/predict-typed, "
                             f"{MAX_NEW_TOKENS} otherwise")
    parser.add_argument("--resume", action="store_true",
                        help="resume training from the last checkpoint in --output-dir")
    args = parser.parse_args()

    if args.max_new_tokens is None:
        long_modes = ("fill-missing", "predict-typed")
        args.max_new_tokens = LONG_MAX_NEW_TOKENS if args.mode in long_modes else MAX_NEW_TOKENS
    if args.mode in ("evaluate", "predict-valid", "predict", "fill-missing", "predict-typed") \
            and not args.adapter_dir:
        parser.error(f"--adapter-dir is required for --mode {args.mode}")
    if args.mode in ("predict-valid", "predict", "fill-missing", "predict-typed") and not args.image_dir:
        parser.error(f"--image-dir is required for --mode {args.mode}")
    if args.mode == "plot" and not args.trainer_state:
        parser.error("--trainer-state is required for --mode plot")
    return args


def main():
    args = parse_args()

    if args.mode == "train":
        train(args)
        return
    if args.mode == "plot":
        os.makedirs(args.output_dir, exist_ok=True)
        plot_losses(args.trainer_state, os.path.join(args.output_dir, "loss_curves.png"))
        return

    device_map = "auto" if args.mode == "predict-typed" else DEVICE
    model, tokenizer, processor = load_finetuned(args.model_id, args.adapter_dir, device_map)
    with torch.no_grad():
        if args.mode == "evaluate":
            evaluate(args, model, processor)
        elif args.mode == "predict-valid":
            predict_valid(args, model, processor)
        elif args.mode == "predict":
            predict(args, model, processor)
        elif args.mode == "fill-missing":
            fill_missing(args, model, tokenizer)
        elif args.mode == "predict-typed":
            predict_typed(args, model, tokenizer)


if __name__ == "__main__":
    main()
