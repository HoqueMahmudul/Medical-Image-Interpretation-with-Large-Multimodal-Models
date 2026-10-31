"""Caption prediction with an adapted MoonDream2 (ImageCLEFmedical caption task).

MoonDream2 (vikhyatk/moondream2, revision 2024-04-02) is a small vision-language
model: a SigLIP vision encoder, an MLP projector and a Phi-1.5 text model. This
script adapts it to medical image captioning:

* ``prepare``  - replace the vision projector with a freshly initialised
                 1152 -> 8192 -> 2048 MLP, cast the model to bfloat16 and save it
                 (this is how ``Mahmudul/moondream_custom`` was produced).
* ``train``    - LoRA fine-tuning of the adapted model on image/caption pairs
                 with a custom training loop (8-bit Adam, cosine LR schedule,
                 per-epoch validation loss, early stopping, W&B logging).
* ``evaluate`` - print ground truth vs. generated caption for the first test
                 samples (qualitative check).
* ``predict``  - generate a caption for every image of the test dataset and
                 write them to a CSV file (image_name, caption).

Usage:
    python moondream2.py --mode prepare --output-dir moondream_custom
    python moondream2.py --mode train --epochs 10 --batch-size 8
    python moondream2.py --mode evaluate --adapter-dir checkpoints-MoonDream/Mahmudul/moondream_custom-ft
    python moondream2.py --mode predict --adapter-dir checkpoints-MoonDream/Mahmudul/moondream_custom-ft

Environment:
    HF_TOKEN        Hugging Face token for gated/private models and datasets
                    (or run ``huggingface-cli login`` once).
    WANDB_API_KEY   Weights & Biases API key, picked up automatically by wandb
                    (or run ``wandb login`` once). Not needed with --no-wandb.

Requirements:
    torch, torchvision, transformers==4.37.2, accelerate==0.21.0, datasets,
    peft, bitsandbytes, einops, timm, sentencepiece==0.1.99, hf_transfer,
    wandb, pandas, tqdm, pillow
"""

import argparse
import math
import os

# Faster model downloads; must be set before the HuggingFace libraries are imported.
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

import pandas as pd
import torch
from bitsandbytes.optim import Adam8bit
from datasets import load_dataset
from peft import LoraConfig, PeftModel, get_peft_model
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# --- Defaults ---

BASE_MODEL_ID = "vikhyatk/moondream2"
CUSTOM_MODEL_ID = "Mahmudul/moondream_custom"
MODEL_REVISION = "2024-04-02"
TRAIN_DATASET_ID = "HoqueMahmudul/clef2024_train_valid_a"
TEST_DATASET_ID = "HoqueMahmudul/test_dataset"
OUTPUT_DIR = "checkpoints-MoonDream"
PREDICTIONS_CSV = "moondream-30epochs.csv"

DEVICE = "cuda"
# CPU doesn't support float16; bfloat16 is preferred on Ampere GPUs.
DTYPE = torch.float32 if DEVICE == "cpu" else torch.bfloat16
USE_4BIT = False

QUESTION = "What do you see?"
ANSWER_EOS = "<|endoftext|>"
IMG_TOKENS = 729  # number of tokens used to represent each image
IMAGE_SIZE = 378

EPOCHS = 10
BATCH_SIZE = 8
EVAL_BATCH_SIZE = 8
GRAD_ACCUM_STEPS = 4
LEARNING_RATE = 3e-5
LORA_ALPHA = 32
LORA_RANK = 64
EARLY_STOPPING_PATIENCE = 6
EARLY_STOPPING_MIN_DELTA = 0.0001
WANDB_PROJECT = "model-ft"
NUM_EVAL_SAMPLES = 3

PREPROCESS = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


# --- Model ---

class AdjustedMMProjector(nn.Module):
    """Two-layer MLP projecting 1152-d vision features to the 2048-d text space."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(in_features=1152, out_features=8192, bias=True)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(in_features=8192, out_features=2048, bias=True)

        nn.init.normal_(self.fc1.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.fc1.bias)
        nn.init.normal_(self.fc2.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        return self.fc2(self.gelu(self.fc1(x)))


class VisionProjection(nn.Module):
    """Wrapper so the new projector sits at ``vision_encoder.projection.mlp``."""

    def __init__(self):
        super().__init__()
        self.mlp = AdjustedMMProjector()

    def forward(self, x):
        return self.mlp(x)


def to_bfloat16(model):
    """Cast every parameter and buffer of the model to bfloat16 in place."""
    for param in model.parameters():
        param.data = param.data.to(torch.bfloat16)
    for buffer in model.buffers():
        buffer.data = buffer.data.to(torch.bfloat16)


def load_model(model_id, revision):
    """Load a MoonDream2 checkpoint and its tokenizer."""
    quantization_config = None
    if USE_4BIT:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=DTYPE,
        )
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        trust_remote_code=True,
        torch_dtype=DTYPE,
        device_map={"": DEVICE},
        quantization_config=quantization_config,
    )
    return model, tokenizer


def load_for_inference(model_id, revision, adapter_dir):
    """Load the adapted model and, if given, the fine-tuned LoRA adapter."""
    model, tokenizer = load_model(model_id, revision)
    if adapter_dir:
        model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    return model, tokenizer


def apply_lora(model):
    """Wrap the model with LoRA adapters on the projector, MLP and attention layers."""
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=["proj", "fc1", "fc2", "Wqkv", "out_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


# --- Data ---

class ImageCLEFDataset(Dataset):
    """Image/caption pairs formatted as a single question-answer turn."""

    def __init__(self, dataset_id, split="train"):
        self.data = load_dataset(dataset_id, split=split)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        image = PREPROCESS(sample["image"].convert("RGB"))
        return {
            "image": image,
            "qa": [{"question": QUESTION, "answer": sample["caption"]}],
        }


class ImageCLEFTestDataset(Dataset):
    """Unlabelled test images together with their file names."""

    def __init__(self, dataset_id, split="test"):
        self.data = load_dataset(dataset_id, split=split)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return {
            "image": PREPROCESS(sample["image"].convert("RGB")),
            "image_name": sample["image_name"],
        }


def make_collate_fn(model, tokenizer):
    """Build a collate function that tokenises Q/A turns and masks non-answer tokens."""

    def collate_fn(batch):
        images = torch.stack([sample["image"] for sample in batch])
        if hasattr(model.vision_encoder, "preprocess"):
            images = model.vision_encoder.preprocess(images)

        labels_acc = []
        tokens_acc = []
        for sample in batch:
            toks = [tokenizer.bos_token_id]
            # The image embeddings are inserted after BOS, so they are masked too.
            labs = [-100] * (IMG_TOKENS + 1)
            for qa in sample["qa"]:
                q_t = tokenizer(
                    f"\n\nQuestion: {qa['question']}\n\nAnswer:", add_special_tokens=False
                ).input_ids
                toks.extend(q_t)
                labs.extend([-100] * len(q_t))
                a_t = tokenizer(
                    f" {qa['answer']}{ANSWER_EOS}", add_special_tokens=False
                ).input_ids
                toks.extend(a_t)
                labs.extend(a_t)
            tokens_acc.append(toks)
            labels_acc.append(labs)

        max_len = max(len(labels) for labels in labels_acc)
        attn_mask_acc = []
        for i in range(len(batch)):
            len_i = len(labels_acc[i])
            pad_i = max_len - len_i
            labels_acc[i].extend([-100] * pad_i)
            tokens_acc[i].extend([tokenizer.eos_token_id] * pad_i)
            attn_mask_acc.append([1] * len_i + [0] * pad_i)

        return (
            images.to(dtype=DTYPE),
            torch.stack([torch.tensor(t, dtype=torch.long) for t in tokens_acc]),
            torch.stack([torch.tensor(lab, dtype=torch.long) for lab in labels_acc]),
            torch.stack([torch.tensor(a, dtype=torch.bool) for a in attn_mask_acc]),
        )

    return collate_fn


# --- Training ---

class EarlyStopping:
    """Stop training when the validation loss stops improving."""

    def __init__(self, patience=2, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


def compute_loss(model, batch):
    """Language-modelling loss with image embeddings spliced in after BOS."""
    images, tokens, labels, attn_mask = (t.to(DEVICE) for t in batch)

    # The vision tower (including the projector) is run without gradients.
    with torch.no_grad():
        img_embs = model.vision_encoder.encoder(images)
        img_embs = model.vision_encoder.projection(img_embs)

    tok_embs = model.text_model.get_input_embeddings()(tokens)
    inputs_embeds = torch.cat((tok_embs[:, 0:1, :], img_embs, tok_embs[:, 1:, :]), dim=1)

    outputs = model.text_model(inputs_embeds=inputs_embeds, labels=labels, attention_mask=attn_mask)
    return outputs.loss


def lr_schedule(step, max_steps, base_lr):
    """Linear warm-up over the first 10% of steps, then cosine decay to 10% of base_lr."""
    x = step / max_steps
    if x < 0.1:
        return 0.1 * base_lr + 0.9 * base_lr * x / 0.1
    return 0.1 * base_lr + 0.9 * base_lr * (1 + math.cos(math.pi * (x - 0.1))) / 2


def train(args):
    """Fine-tune the adapted MoonDream2 with LoRA and save the adapter."""
    model, tokenizer = load_model(args.model_id, args.revision)
    model = apply_lora(model)

    datasets = {
        "train": ImageCLEFDataset(args.dataset_id, "train"),
        "test": ImageCLEFDataset(args.dataset_id, "test"),
    }
    collate_fn = make_collate_fn(model, tokenizer)
    dataloaders = {
        "train": DataLoader(
            datasets["train"], batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn
        ),
        "test": DataLoader(datasets["test"], batch_size=EVAL_BATCH_SIZE, collate_fn=collate_fn),
    }
    total_steps = args.epochs * len(dataloaders["train"]) // args.grad_accum_steps

    lr_scaling = LORA_ALPHA / (LORA_RANK ** 0.5)
    print("Using an LR scaling for LoRA adapters of: ", lr_scaling)

    model.text_model.train()
    # Non-reentrant checkpointing avoids "no grad" errors with LoRA.
    model.text_model.transformer.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    lora_params = []
    for name, module in model.named_modules():
        if "lora" in name:
            lora_params.extend([p for p in module.parameters() if p.requires_grad])

    optimizer = Adam8bit(
        [{"params": lora_params}],
        lr=args.learning_rate * 0.1,
        betas=(0.9, 0.95),
        eps=1e-6,
    )

    if args.use_wandb:
        import wandb

        wandb.init(
            project=WANDB_PROJECT,
            config={
                "EPOCHS": args.epochs,
                "BATCH_SIZE": args.batch_size,
                "GRAD_ACCUM_STEPS": args.grad_accum_steps,
                "LR": args.learning_rate,
            },
        )

    early_stopper = EarlyStopping(patience=EARLY_STOPPING_PATIENCE, min_delta=EARLY_STOPPING_MIN_DELTA)
    step = 0
    for epoch in range(args.epochs):
        for batch in tqdm(dataloaders["train"], desc=f"Epoch {epoch + 1}/{args.epochs}"):
            step += 1
            loss = compute_loss(model, batch)
            loss.backward()

            if step % args.grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            lr = lr_schedule(step / args.grad_accum_steps, total_steps, args.learning_rate)
            for param_group in optimizer.param_groups:
                if param_group["params"] == lora_params:
                    param_group["lr"] = lr * lr_scaling
                else:
                    param_group["lr"] = lr

        val_loss = 0
        for val_batch in tqdm(dataloaders["test"], desc="Validation"):
            with torch.no_grad():
                val_loss += compute_loss(model, val_batch).item()
        val_loss /= len(dataloaders["test"])

        if args.use_wandb:
            wandb.log({
                "epoch": epoch,
                "loss/train": loss.item(),
                "loss/val": val_loss,
                "lr": optimizer.param_groups[0]["lr"],
            })

        early_stopper(val_loss)
        if early_stopper.early_stop:
            print("Early stopping triggered")
            break

    if args.use_wandb:
        wandb.finish()

    save_dir = os.path.join(args.output_dir, f"{args.model_id}-ft")
    model.save_pretrained(save_dir)
    print(f"Adapter saved to {save_dir}")


# --- Projector replacement ---

def prepare(args):
    """Swap in the new vision projector, cast to bfloat16 and save the model."""
    model, tokenizer = load_model(args.model_id, args.revision)
    model.vision_encoder.projection = VisionProjection()
    if torch.cuda.is_available():
        model.vision_encoder.projection.to("cuda")
    to_bfloat16(model)

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Adapted model saved to {args.output_dir}")
    if args.push_to_hub:
        model.push_to_hub(args.push_to_hub, private=True)
        tokenizer.push_to_hub(args.push_to_hub, private=True)


# --- Evaluation ---

def evaluate(args):
    """Print ground truth and model answer for the first few test samples."""
    model, tokenizer = load_for_inference(args.model_id, args.revision, args.adapter_dir)
    test_dataset = ImageCLEFDataset(args.dataset_id, "test")

    with torch.no_grad():
        for i in range(min(NUM_EVAL_SAMPLES, len(test_dataset))):
            sample = test_dataset[i]
            encoded_image = model.encode_image(sample["image"].unsqueeze(0))
            for qa in sample["qa"]:
                print("Question:", qa["question"])
                print("Ground Truth:", qa["answer"])
                answer = model.answer_question(encoded_image, qa["question"], tokenizer=tokenizer)
                print("Model answer:", answer)


# --- Inference ---

def generate_captions(data_loader, model, tokenizer):
    """Caption every image in the loader and return a list of result dicts."""
    results = []
    with torch.no_grad():
        for data in tqdm(data_loader, desc="Captioning"):
            images, image_names = data["image"], data["image_name"]
            encoded_images = [model.encode_image(image.unsqueeze(0)) for image in images]
            # Generation settings come from MoonDream's own answer_question().
            captions = [
                model.answer_question(encoded_image, QUESTION, tokenizer=tokenizer)
                for encoded_image in encoded_images
            ]
            for image_name, caption in zip(image_names, captions):
                print(f"Image Name: {image_name}")
                print(f"Generated Caption: {caption}")
                results.append({"image_name": image_name, "caption": caption})
    return results


def predict(args):
    """Generate captions for the test dataset and write them to CSV."""
    model, tokenizer = load_for_inference(args.model_id, args.revision, args.adapter_dir)
    test_dataset = ImageCLEFTestDataset(args.test_dataset_id, "test")
    data_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    captions_df = pd.DataFrame(generate_captions(data_loader, model, tokenizer))
    print(captions_df.head())

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, args.predictions_csv)
    captions_df.to_csv(out_path, index=False)
    print(f"Captions saved to {out_path}")


# --- CLI ---

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--mode", choices=["prepare", "train", "evaluate", "predict"], required=True)
    parser.add_argument("--model-id", default=None,
                        help=f"default: {BASE_MODEL_ID} for prepare, {CUSTOM_MODEL_ID} otherwise")
    parser.add_argument("--revision", default=MODEL_REVISION)
    parser.add_argument("--dataset-id", default=TRAIN_DATASET_ID,
                        help="train/test captioned dataset (train, evaluate)")
    parser.add_argument("--test-dataset-id", default=TEST_DATASET_ID,
                        help="unlabelled test dataset with image_name column (predict)")
    parser.add_argument("--output-dir", default=None,
                        help=f"default: moondream_custom for prepare, {OUTPUT_DIR} otherwise")
    parser.add_argument("--adapter-dir", default=None,
                        help="LoRA adapter saved by --mode train (evaluate, predict)")
    parser.add_argument("--predictions-csv", default=PREDICTIONS_CSV)
    parser.add_argument("--push-to-hub", default=None, metavar="REPO_ID",
                        help="also push the adapted model to this private Hub repo (prepare)")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--grad-accum-steps", type=int, default=GRAD_ACCUM_STEPS)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--no-wandb", dest="use_wandb", action="store_false",
                        help="disable Weights & Biases logging")
    args = parser.parse_args()

    if args.model_id is None:
        args.model_id = BASE_MODEL_ID if args.mode == "prepare" else CUSTOM_MODEL_ID
    if args.output_dir is None:
        args.output_dir = "moondream_custom" if args.mode == "prepare" else OUTPUT_DIR
    return args


def main():
    args = parse_args()
    if args.mode == "prepare":
        prepare(args)
    elif args.mode == "train":
        train(args)
    elif args.mode == "evaluate":
        evaluate(args)
    elif args.mode == "predict":
        predict(args)


if __name__ == "__main__":
    main()
