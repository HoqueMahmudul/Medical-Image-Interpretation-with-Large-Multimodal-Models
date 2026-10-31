"""Caption prediction with SmolVLM-500M-Instruct (ImageCLEFmedical Caption 2025).

SmolVLM-500M-Instruct is a compact Idefics3-based vision-language model. This
script LoRA fine-tunes it on medical image/caption pairs and generates captions
for the ImageCLEFmedical 2025 test set.

Modes:
* ``clean-data``     - re-encode images with broken EXIF/XMP metadata or
                       truncated data as PNG and save the split to disk
                       (run once for the train split and once for the test split).
* ``train``          - LoRA fine-tuning with the Hugging Face Trainer on the
                       cleaned splits; loss on the caption tokens only.
* ``preview``        - print captions for the first few training images.
* ``predict``        - caption every image of the test dataset into a CSV.
* ``predict-typed``  - caption the test dataset and prefix the image type
                       predicted by a separate classifier (CSV with ID,
                       predicted_label).
* ``plot``           - plot training/validation loss from a trainer_state.json.

Usage:
    python smolvlm.py --mode clean-data --dataset-id racho1/2025 --split train --data-dir Clef_2025_clean
    python smolvlm.py --mode clean-data --dataset-id racho1/Clef_2025 --split test --data-dir Clef_test_2025_clean
    python smolvlm.py --mode train --output-dir SMOLVLM
    python smolvlm.py --mode predict --adapter-dir SMOLVLM/checkpoint-5004
    python smolvlm.py --mode predict-typed --adapter-dir SMOLVLM/checkpoint-5004 \
        --classification-csv test_predictions_Final.csv
    python smolvlm.py --mode plot --trainer-state SMOLVLM/checkpoint-5004/trainer_state.json

Environment:
    HF_TOKEN   Hugging Face token for gated/private models and datasets
               (or run ``huggingface-cli login`` once).
    Training logs go to every installed reporting integration (e.g. wandb, in
    which case set WANDB_API_KEY or run ``wandb login``).

Requirements:
    torch, torchvision, transformers, accelerate, datasets, peft, bitsandbytes,
    hf_transfer, tensorboard, ninja, flash_attn==2.7.3, pandas, pillow, tqdm,
    matplotlib
"""

import argparse
import csv
import io
import json
import os
from itertools import islice

# Faster model downloads; must be set before the HuggingFace libraries are imported.
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

import pandas as pd
import torch
from datasets import Image as HFImage
from datasets import load_dataset, load_from_disk
from PIL import Image, ImageFile, UnidentifiedImageError
from peft import LoraConfig, PeftModel, get_peft_model
from tqdm import tqdm, trange
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
    EarlyStoppingCallback,
    Idefics3ForConditionalGeneration,
    Trainer,
    TrainingArguments,
)

ImageFile.LOAD_TRUNCATED_IMAGES = True  # tolerate truncated JPEG bit-streams

# --- Defaults ---

MODEL_ID = "HuggingFaceTB/SmolVLM-500M-Instruct"
TRAIN_DATA_DIR = "Clef_2025_clean"
EVAL_DATA_DIR = "Clef_test_2025_clean"
TEST_DATASET_ID = "racho1/test_dataset"
TEST_SPLIT = "test"
OUTPUT_DIR = "SMOLVLM"
DEVICE = "cuda"

TRAIN_QUESTION = "What do you see here?"
PREVIEW_PROMPT = "What does this medical image show?"
PREDICT_PROMPT = "Describe the medical image."

EPOCHS = 2
BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 16  # effective batch size 32
LEARNING_RATE = 5e-5
WARMUP_STEPS = 500
MAX_NEW_TOKENS = 64
NUM_PREVIEW_SAMPLES = 5
VERIFY_ROWS = {"train": 10_000, "test": 17_000}  # rows decoded after cleaning

MODALITIES = ["MRI", "CT", "Radiograph", "Ultrasound"]


# --- Data ---

def is_bad_image(raw):
    """Return True if the image bytes have unreadable metadata or truncated data."""
    try:
        with Image.open(io.BytesIO(raw)) as im:
            im.getexif()  # raises UnicodeDecodeError on bad XMP
            im.load()     # detects truncated data
        return False
    except (UnicodeDecodeError, UnidentifiedImageError, OSError):
        return True


def clean_dataset(dataset_id, split, out_dir):
    """Re-encode bad images of one split as PNG (no EXIF/XMP) and save to disk."""
    print("Loading dataset (decode=False)...")
    ds = load_dataset(dataset_id, split=split, streaming=False)
    ds = ds.cast_column("image", HFImage(decode=False))

    print("Scanning rows...")
    bad_ids = [i for i in trange(len(ds), desc="scan") if is_bad_image(ds[i]["image"]["bytes"])]
    print(f"   found {len(bad_ids)} bad rows")
    bad_set = set(bad_ids)

    def repair(example, idx):
        if idx in bad_set:
            with Image.open(io.BytesIO(example["image"]["bytes"])) as im:
                buf = io.BytesIO()
                im.convert("RGB").save(buf, format="PNG")
            example["image"]["bytes"] = buf.getvalue()
        return example

    print("Re-encoding bad rows...")
    clean_ds = ds.map(repair, with_indices=True, desc="repair")

    print(f"Saving cleaned dataset to {out_dir}...")
    clean_ds.save_to_disk(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "clean_info.json"), "w") as f:
        json.dump({
            "source": dataset_id,
            "split": split,
            "total_rows": len(ds),
            "bad_rows_fixed": len(bad_ids),
        }, f, indent=2)

    n_verify = VERIFY_ROWS.get(split, 10_000)
    print(f"Verifying first {n_verify} rows...")
    ver_ds = clean_ds.cast_column("image", HFImage())
    for i in trange(min(n_verify, len(ver_ds))):
        _ = ver_ds[i]["image"].size  # raises if the image is still broken
    print("Verification passed")


def load_clean_split(path):
    """Load a split saved by clean-data with images decoded."""
    return load_from_disk(path).cast_column("image", HFImage())


class AssistantOnlyCollator:
    """Build chat-formatted batches and mask everything up to the "Assistant:" header."""

    def __init__(self, processor):
        self.processor = processor
        self.assist_tokens = processor.tokenizer(
            "\nAssistant:", add_special_tokens=False, return_tensors="pt"
        )["input_ids"][0]

    def __call__(self, examples):
        texts, images = [], []
        for ex in examples:
            messages = [
                {"role": "user", "content": [
                    {"type": "text", "text": TRAIN_QUESTION},
                    {"type": "image"},
                ]},
                {"role": "assistant", "content": [{"type": "text", "text": ex["caption"]}]},
            ]
            text = self.processor.apply_chat_template(messages, add_generation_prompt=False)
            texts.append(text.strip())
            images.append(ex["image"])

        batch = self.processor(text=texts, images=images, return_tensors="pt", padding=True)
        labels = batch["input_ids"].clone()
        for i, input_ids in enumerate(batch["input_ids"]):
            start = self._find_subseq(input_ids, self.assist_tokens)
            if start is not None:
                labels[i, : start + len(self.assist_tokens)] = -100
            else:
                labels[i, :] = -100  # no assistant header found: no loss from this sample
        batch["labels"] = labels
        return batch

    @staticmethod
    def _find_subseq(sequence, subseq):
        """Return the start index of subseq in sequence, or None."""
        n, m = len(sequence), len(subseq)
        for idx in range(n - m + 1):
            if torch.equal(sequence[idx: idx + m], subseq):
                return idx
        return None


# --- Model ---

def load_base_model(model_id):
    """Load SmolVLM (Idefics3) in bfloat16 with flash attention on CUDA."""
    model = Idefics3ForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        _attn_implementation="flash_attention_2" if DEVICE == "cuda" else "eager",
    )
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor


def load_finetuned(model_id, adapter_dir):
    """Load the base model with a trained LoRA adapter for inference."""
    base_model = AutoModelForVision2Seq.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, adapter_dir).eval()
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor


def apply_lora(model):
    """Add LoRA adapters to every linear layer; train lm_head and embeddings fully."""
    lora_config = LoraConfig(
        r=64,
        lora_alpha=16,
        use_rslora=True,
        target_modules="all-linear",
        modules_to_save=["lm_head", "embed_tokens"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


# --- Training ---

def train(args):
    """LoRA fine-tune SmolVLM on the cleaned caption splits."""
    model, processor = load_base_model(args.model_id)
    train_dataset = load_clean_split(args.train_data_dir)
    eval_dataset = load_clean_split(args.eval_data_dir)

    model = apply_lora(model)
    model.gradient_checkpointing_enable()  # trades compute for memory
    model.enable_input_require_grads()     # needed for checkpointing with PEFT
    model.config.use_cache = False

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        run_name="smolvlm_medical_lora_ep2",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=WARMUP_STEPS,
        bf16=True,
        remove_unused_columns=False,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=5,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_steps=100,
        dataloader_num_workers=8,
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=AssistantOnlyCollator(processor),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)


# --- Inference ---

def caption_image(model, processor, pil_img, prompt, max_new_tokens, empty_placeholder=None):
    """Generate a caption for one image (optionally a placeholder if nothing is generated)."""
    messages = [{"role": "user", "content": [
        {"type": "text", "text": prompt},
        {"type": "image"},
    ]}]
    chat = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=[chat], images=[pil_img], return_tensors="pt", padding=True).to(DEVICE)

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens)

    start = inputs["input_ids"].shape[1]
    if empty_placeholder is not None and output.shape[1] <= start:
        return empty_placeholder
    return processor.decode(output[0, start:], skip_special_tokens=True).strip()


def preview(args):
    """Print captions for the first few images of the training split."""
    model, processor = load_finetuned(args.model_id, args.adapter_dir) if args.adapter_dir \
        else load_base_model(args.model_id)
    model.eval()
    dataset = load_clean_split(args.train_data_dir)
    for example in islice(dataset, NUM_PREVIEW_SAMPLES):
        print("Prediction:", caption_image(model, processor, example["image"], PREVIEW_PROMPT,
                                           args.max_new_tokens), "\n")


def write_csv(path, header, rows):
    """Write rows to a CSV file, creating its folder if needed."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def predict(args):
    """Caption every image of the test dataset and save image_id, caption pairs."""
    model, processor = load_finetuned(args.model_id, args.adapter_dir)
    test_ds = load_dataset(
        args.test_dataset_id, split=TEST_SPLIT, streaming=True, verification_mode="no_checks"
    ).cast_column("image", HFImage())

    results = []
    print("Generating captions...")
    for idx, ex in enumerate(test_ds):
        # The test dataset has an "image_name" column, not "id"/"file_name",
        # so the IDs written here are "image_<idx>" (as in the submitted run).
        img_id = ex.get("id") or ex.get("file_name") or f"image_{idx}"
        try:
            caption = caption_image(model, processor, ex["image"], PREDICT_PROMPT, args.max_new_tokens)
            print(f"{img_id[:30]} -> {caption}")
            results.append((img_id, caption))
        except Exception as e:
            print(f"Failed on {img_id}: {e}")

    out_path = os.path.join(args.output_dir, "test_predictions.csv")
    write_csv(out_path, ["image_id", "generated_caption"], results)
    print(f"\nCaptions saved to {out_path} (Total: {len(results)} images)")


def remove_hallucinated_modality(caption):
    """Drop a leading modality word (and a following "scan"/"image") from a caption."""
    words = caption.strip().split()
    if not words:
        return caption
    if words[0].lower() in [m.lower() for m in MODALITIES]:
        words = words[1:]
        if words and words[0].lower() in ["scan", "image"]:
            words = words[1:]
    return " ".join(words).strip()


def predict_typed(args):
    """Caption the test dataset and prefix each caption with the classifier's image type."""
    df_class = pd.read_csv(args.classification_csv)
    df_class = df_class.rename(columns={"ID": "image_id", "predicted_label": "type"})
    df_class["image_id"] = df_class["image_id"].astype(str).str.strip() + ".jpg"
    df_class["type"] = df_class["type"].astype(str).str.strip().str.capitalize()
    type_map = dict(zip(df_class["image_id"], df_class["type"]))

    dataset = load_dataset(
        args.test_dataset_id, split=TEST_SPLIT, streaming=False, verification_mode="no_checks"
    ).cast_column("image", HFImage())
    model, processor = load_finetuned(args.model_id, args.adapter_dir)

    results = []
    print("Generating captions...")
    for idx, example in enumerate(tqdm(dataset)):
        # NOTE: assumes row idx of the dataset is test image number idx; the
        # dataset's own "image_name" column is not used for matching.
        image_id = f"ImageCLEFmedical_Caption_2025_test_{idx}.jpg"
        if image_id not in type_map:
            print(f"Skipping unknown image_id: {image_id}")
            continue

        label = type_map[image_id]
        try:
            generated = caption_image(
                model, processor, example["image"], PREDICT_PROMPT, args.max_new_tokens,
                empty_placeholder="[No caption generated]",
            )
            final_caption = f"{label} showing {remove_hallucinated_modality(generated)}"
            print(f"{image_id} -> {final_caption}")
            results.append((image_id, label, final_caption))
        except Exception as e:
            print(f"Failed on {image_id}: {e}")

    out_path = os.path.join(args.output_dir, "newtest_predictions.csv")
    write_csv(out_path, ["image_id", "type", "generated_caption"], results)
    print(f"\nCaptions saved to {out_path} (Total: {len(results)} images)")


# --- Evaluation ---

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
        "clean-data", "train", "preview", "predict", "predict-typed", "plot",
    ])
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--dataset-id", default="racho1/2025",
                        help="Hub dataset to clean (clean-data)")
    parser.add_argument("--split", default="train", help="split to clean (clean-data)")
    parser.add_argument("--data-dir", default=TRAIN_DATA_DIR,
                        help="where clean-data saves the cleaned split")
    parser.add_argument("--train-data-dir", default=TRAIN_DATA_DIR)
    parser.add_argument("--eval-data-dir", default=EVAL_DATA_DIR)
    parser.add_argument("--test-dataset-id", default=TEST_DATASET_ID)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--adapter-dir", default=None,
                        help="LoRA checkpoint, e.g. SMOLVLM/checkpoint-5004")
    parser.add_argument("--resume-from-checkpoint", default=None,
                        help="checkpoint to resume training from, e.g. SMOLVLM/checkpoint-2503")
    parser.add_argument("--classification-csv", default="test_predictions_Final.csv",
                        help="CSV with ID and predicted_label columns (predict-typed)")
    parser.add_argument("--trainer-state", default=None, help="trainer_state.json to plot (plot)")
    parser.add_argument("--epochs", type=float, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--grad-accum-steps", type=int, default=GRAD_ACCUM_STEPS)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    args = parser.parse_args()

    if args.mode in ("predict", "predict-typed") and not args.adapter_dir:
        parser.error(f"--adapter-dir is required for --mode {args.mode}")
    if args.mode == "plot" and not args.trainer_state:
        parser.error("--trainer-state is required for --mode plot")
    return args


def main():
    args = parse_args()
    if args.mode == "clean-data":
        clean_dataset(args.dataset_id, args.split, args.data_dir)
    elif args.mode == "train":
        train(args)
    elif args.mode == "preview":
        preview(args)
    elif args.mode == "predict":
        predict(args)
    elif args.mode == "predict-typed":
        predict_typed(args)
    elif args.mode == "plot":
        os.makedirs(args.output_dir, exist_ok=True)
        plot_losses(args.trainer_state, os.path.join(args.output_dir, "loss_curves.png"))


if __name__ == "__main__":
    main()
