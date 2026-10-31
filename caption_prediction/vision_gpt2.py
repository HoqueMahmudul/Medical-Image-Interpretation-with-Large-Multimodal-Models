"""VisionGPT2: ViT-B/16 encoder + GPT-2 decoder with cross-attention for caption prediction.

The model interleaves the 12 blocks of a pretrained ``vit_base_patch16_224`` (timm)
with 12 GPT-2 blocks (weights from HuggingFace ``gpt2``). Each GPT-2 block gets an
extra, newly initialised cross-attention layer that attends to the ViT tokens of the
same depth. Training is staged: only the cross-attention layers are trained in
epoch 0, GPT-2 is unfrozen at epoch 1 and everything is unfrozen at epoch 2.

Modes:
    train    Train on ImageCLEFmedical 2024 (ROCOv2) captions, keep the checkpoint with
             the best validation perplexity, save metrics and loss/perplexity plots.
    sample   Caption a few random validation images and save them as figures.
    predict  Caption every .jpg/.png in the test folder (greedy decoding) and write a CSV.

Data layout (ImageCLEFmedical 2024 caption task):
    train_captions.csv / valid_captions.csv with columns ``ID`` and ``Caption``;
    images stored as ``<images-dir>/<ID>.jpg``.

Usage:
    python vision_gpt2.py --mode train --valid-csv valid_captions.csv --valid-images ../Valid/
    python vision_gpt2.py --mode predict --test-images ../Test/ --output-dir captioner

Note: like the original notebook, the default training split is carved out of the
*validation* CSV (90/10 split). Use ``--split-source train`` to split the training CSV.

Environment:
    Pretrained weights are downloaded from the HuggingFace Hub (``gpt2``) and timm.
    For gated/private access set HF_TOKEN or run ``huggingface-cli login``.

Requirements:
    torch, timm, transformers, albumentations, numpy, pandas, scikit-learn,
    pillow, matplotlib, tqdm
"""

import argparse
import gc
import os
from pathlib import Path
from types import SimpleNamespace

import albumentations as A
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from PIL import Image
from sklearn.model_selection import train_test_split
from timm import create_model
from torch.cuda.amp import GradScaler, autocast
from tqdm.auto import tqdm
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# --- Defaults ---

IMAGE_SIZE = 224  # fixed by the pretrained vit_base_patch16_224 position embeddings
NORM_MEAN = [0.5, 0.5, 0.5]
NORM_STD = [0.5, 0.5, 0.5]
TEST_EXTENSIONS = ("jpg", "png")

MODEL_CONFIG = SimpleNamespace(
    vocab_size=50_257,
    embed_dim=768,
    num_heads=12,
    seq_len=1024,
    depth=12,
    attention_dropout=0.1,
    residual_dropout=0.1,
    mlp_ratio=4,
    mlp_dropout=0.1,
    emb_dropout=0.1,
)

DEFAULT_EPOCHS = 100
DEFAULT_FREEZE_EPOCHS_GPT = 1
DEFAULT_FREEZE_EPOCHS_ALL = 2
DEFAULT_LR = 1e-4
DEFAULT_BATCH_SIZE = 4
DEFAULT_OUTPUT_DIR = "captioner"
CHECKPOINT_NAME = "captioner.pt"


# --- Data ---

def get_transforms():
    """Return (train, valid) albumentations pipelines."""
    sample_tfms = [
        A.HorizontalFlip(),
        A.RandomBrightnessContrast(),
        A.ColorJitter(),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.3, rotate_limit=45, p=0.5),
        A.HueSaturationValue(p=0.3),
    ]
    train_tfms = A.Compose([
        *sample_tfms,
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(mean=NORM_MEAN, std=NORM_STD, always_apply=True),
        ToTensorV2(),
    ])
    valid_tfms = A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(mean=NORM_MEAN, std=NORM_STD, always_apply=True),
        ToTensorV2(),
    ])
    return train_tfms, valid_tfms


def load_tokenizer():
    """GPT-2 tokenizer with the EOS token reused for padding."""
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_caption_df(csv_path, images_dir):
    """Read an ID/Caption CSV and turn IDs into image paths."""
    df = pd.read_csv(csv_path)
    df = df.rename(columns={"ID": "image", "Caption": "caption"})
    df["image"] = images_dir + df["image"].astype(str) + ".jpg"
    return df


class CaptionDataset:
    """Image/caption pairs tokenised for next-token prediction."""

    def __init__(self, df, tfms, tokenizer):
        self.df = df
        self.tfms = tfms
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sample = self.df.iloc[idx, :]
        image = np.array(Image.open(sample["image"]).convert("RGB"))
        image = self.tfms(image=image)["image"]
        caption = f"{sample['caption']}<|endoftext|>"
        input_ids = self.tokenizer(caption, truncation=True)["input_ids"]
        labels = input_ids.copy()
        labels[:-1] = input_ids[1:]
        return image, input_ids, labels


class CaptionCollator:
    """Pad a batch to the longest caption; padded label positions become -100."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        image = torch.stack([b[0] for b in batch], dim=0)
        input_ids = self._pad([b[1] for b in batch])
        labels = self._pad([b[2] for b in batch])
        mask = (input_ids != self.tokenizer.pad_token_id).long()
        labels[mask == 0] = -100
        return image, input_ids, labels

    def _pad(self, ids):
        return self.tokenizer.pad(
            {"input_ids": ids},
            padding="longest",
            return_attention_mask=False,
            return_tensors="pt",
        )["input_ids"]


def make_dataloaders(train_df, val_df, tokenizer, batch_size):
    """Build the train/validation DataLoaders used by the notebook."""
    train_tfms, valid_tfms = get_transforms()
    collate_fn = CaptionCollator(tokenizer)
    loader_kwargs = dict(
        batch_size=batch_size,
        pin_memory=True,
        num_workers=2,
        persistent_workers=True,
        collate_fn=collate_fn,
    )
    train_dl = torch.utils.data.DataLoader(
        CaptionDataset(train_df, train_tfms, tokenizer), shuffle=True, **loader_kwargs
    )
    val_dl = torch.utils.data.DataLoader(
        CaptionDataset(val_df, valid_tfms, tokenizer), shuffle=False, **loader_kwargs
    )
    return train_dl, val_dl


# --- Model ---

class GPT2Attention(nn.Module):
    """Causal multi-head self-attention (GPT-2 layout, fused qkv projection)."""

    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.embed_dim
        self.n_heads = config.num_heads
        assert self.embed_dim % self.n_heads == 0, "embedding dimension must be divisible by number of heads"
        self.head_size = self.embed_dim // self.n_heads
        self.seq_len = config.seq_len

        self.c_attn = nn.Linear(self.embed_dim, self.head_size * self.n_heads * 3, bias=True)
        self.scale = self.head_size ** -0.5
        self.register_buffer("mask", torch.tril(torch.ones(1, 1, self.seq_len, self.seq_len)))
        self.c_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.attn_dropout = nn.Dropout(config.attention_dropout)
        self.resid_dropout = nn.Dropout(config.residual_dropout)

    def forward(self, x):
        b, t, c = x.shape
        q, k, v = self.c_attn(x).chunk(3, dim=-1)
        # batch x n_heads x seq_len x head_size
        q = q.view(b, t, self.n_heads, self.head_size).permute(0, 2, 1, 3)
        k = k.view(b, t, self.n_heads, self.head_size).permute(0, 2, 1, 3)
        v = v.view(b, t, self.n_heads, self.head_size).permute(0, 2, 1, 3)

        qk_t = (q @ k.transpose(-2, -1)) * self.scale
        qk_t = qk_t.masked_fill(self.mask[:, :, :t, :t] == 0, float("-inf"))
        weights = self.attn_dropout(F.softmax(qk_t, dim=-1))

        attention = weights @ v
        attention = attention.permute(0, 2, 1, 3).contiguous().view(b, t, c)
        return self.resid_dropout(self.c_proj(attention))


class GPT2CrossAttention(nn.Module):
    """Multi-head cross-attention from caption tokens to image tokens (newly initialised)."""

    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.embed_dim
        self.n_heads = config.num_heads
        assert self.embed_dim % self.n_heads == 0, "embedding dimension must be divisible by number of heads"
        self.head_size = self.embed_dim // self.n_heads
        self.seq_len = config.seq_len

        self.q = nn.Linear(self.embed_dim, self.embed_dim)
        self.k = nn.Linear(self.embed_dim, self.embed_dim)
        self.v = nn.Linear(self.embed_dim, self.embed_dim)
        self.scale = self.head_size ** -0.5
        self.c_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.attn_dropout = nn.Dropout(config.attention_dropout)
        self.resid_dropout = nn.Dropout(config.residual_dropout)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, q, k, v):
        b, t, c = q.shape
        q = self.q(q)
        k = self.k(k)
        v = self.v(v)

        q = q.view(b, q.size(1), self.n_heads, self.head_size).permute(0, 2, 1, 3)
        k = k.view(b, k.size(1), self.n_heads, self.head_size).permute(0, 2, 1, 3)
        v = v.view(b, v.size(1), self.n_heads, self.head_size).permute(0, 2, 1, 3)

        qk_t = (q @ k.transpose(-2, -1)) * self.scale
        weights = self.attn_dropout(F.softmax(qk_t, dim=-1))

        attention = weights @ v
        attention = attention.permute(0, 2, 1, 3).contiguous().view(b, t, c)
        return self.resid_dropout(self.c_proj(attention))


class GPT2MLP(nn.Module):
    """GPT-2 feed-forward block."""

    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.embed_dim
        self.mlp_ratio = config.mlp_ratio
        self.c_fc = nn.Linear(self.embed_dim, self.embed_dim * self.mlp_ratio)
        self.c_proj = nn.Linear(self.embed_dim * self.mlp_ratio, self.embed_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(config.mlp_dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(self.act(self.c_fc(x))))


class GPT2Block(nn.Module):
    """Pre-norm GPT-2 block with an extra cross-attention sub-layer."""

    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.embed_dim
        self.ln_1 = nn.LayerNorm(self.embed_dim)
        self.attn = GPT2Attention(config)
        self.ln_2 = nn.LayerNorm(self.embed_dim)
        self.mlp = GPT2MLP(config)
        self.ln_3 = nn.LayerNorm(self.embed_dim)
        self.cross_attn = GPT2CrossAttention(config)

    def forward(self, x, enc_out):
        x = x + self.attn(self.ln_1(x))
        x = x + self.cross_attn(self.ln_2(x), enc_out, enc_out)
        x = x + self.mlp(self.ln_3(x))
        return x


class VisionGPT2Model(nn.Module):
    """ViT encoder blocks interleaved with GPT-2 decoder blocks."""

    def __init__(self, config):
        super().__init__()
        self.config = config

        vit = create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
        self.patch_embed = vit.patch_embed
        self.cls_token = vit.cls_token
        self.pos_embed = vit.pos_embed
        self.pos_drop = nn.Dropout(p=0.0)
        self.blocks = nn.ModuleList([vit.blocks[i] for i in range(config.depth)])

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.embed_dim),
            wpe=nn.Embedding(config.seq_len, config.embed_dim),
            drop=nn.Dropout(config.emb_dropout),
            h=nn.ModuleList([GPT2Block(config) for _ in range(config.depth)]),
            ln_f=nn.LayerNorm(config.embed_dim),
        ))
        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # weight tying, as in GPT-2

    def _pos_embed(self, x):
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.pos_embed
        return self.pos_drop(x)

    def _gpt_sublayers(self):
        layers = []
        for i in range(self.config.depth):
            block = self.transformer.h[i]
            layers.extend([block.ln_1, block.ln_2, block.attn, block.mlp])
        return layers

    @staticmethod
    def _set_trainable(layers, trainable):
        for layer in layers:
            if isinstance(layer, nn.Parameter):
                layer.requires_grad = trainable
            else:
                for p in layer.parameters():
                    p.requires_grad = trainable

    def pretrained_layers_trainable(self, trainable=False):
        """Freeze/unfreeze every pretrained (ViT + GPT-2) layer; cross-attention is untouched."""
        layers = [
            self.cls_token, self.patch_embed, self.pos_embed, self.blocks,
            self.transformer.wte, self.transformer.wpe,
            self.transformer.ln_f, self.lm_head,
        ]
        layers.extend(self._gpt_sublayers())
        self._set_trainable(layers, trainable)

        total_frozen_params = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        print(f"{total_frozen_params=}")

    def unfreeze_gpt_layers(self):
        """Unfreeze the GPT-2 block sub-layers."""
        self._set_trainable(self._gpt_sublayers(), True)

    @classmethod
    def from_pretrained(cls, config):
        """Build the model and copy HuggingFace GPT-2 weights into the decoder."""
        model = cls(config)
        sd = model.state_dict()
        # ViT weights and the new cross-attention layers are not taken from GPT-2.
        ignore_matches = ["blocks.", "cross_attn.", "ln_3", "cls_token", "pos_embed", "patch_embed.", ".attn.mask"]

        sd_hf = GPT2LMHeadModel.from_pretrained("gpt2").state_dict()
        hf_keys = [
            k for k in sd_hf.keys()
            if not k.endswith(".attn.masked_bias") and not k.endswith(".attn.bias")
        ]
        # HF GPT-2 uses Conv1D, whose weights are transposed relative to nn.Linear.
        transposed = ["attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight"]

        for k in hf_keys:
            if any(match in k for match in ignore_matches):
                continue
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        model.load_state_dict(sd)
        return model

    def forward(self, image, input_ids, labels=None):
        image = self._pos_embed(self.patch_embed(image))

        token_embeddings = self.transformer.wte(input_ids)
        pos_embs = torch.arange(0, input_ids.size(1)).to(input_ids.device)
        positional_embeddings = self.transformer.wpe(pos_embs)
        x = self.transformer.drop(token_embeddings + positional_embeddings)

        for i in range(self.config.depth):
            image = self.blocks[i](image)
            x = self.transformer.h[i](x, image)

        x = self.transformer.ln_f(x)

        if labels is not None:
            lm_logits = self.lm_head(x)
            return F.cross_entropy(lm_logits.view(-1, lm_logits.shape[-1]), labels.view(-1))

        return self.lm_head(x[:, [-1], :])

    def generate(self, image, sequence, eos_token_id, max_tokens=50, temperature=1.0, deterministic=False):
        """Autoregressively extend `sequence`; greedy if deterministic, else multinomial sampling."""
        for _ in range(max_tokens):
            out = self(image, sequence)
            out = out[:, -1, :] / temperature
            probs = F.softmax(out, dim=-1)
            if deterministic:
                next_token = torch.argmax(probs, dim=-1, keepdim=True)
            else:
                next_token = torch.multinomial(probs, num_samples=1)
            sequence = torch.cat([sequence, next_token], dim=1)
            if next_token.item() == eos_token_id:
                break
        return sequence.cpu().flatten()


# --- Training ---

class Trainer:
    """Staged fine-tuning with mixed precision and a OneCycle LR schedule."""

    def __init__(self, model_config, train_config, dls=None):
        self.train_config = train_config
        self.model_config = model_config
        self.device = train_config.device

        self.model = VisionGPT2Model.from_pretrained(model_config).to(self.device)
        self.model.pretrained_layers_trainable(trainable=False)
        print(f"trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}")

        self.tokenizer = load_tokenizer()
        self.gen_tfms = get_transforms()[1]
        self.metrics = pd.DataFrame(columns=["train_loss", "train_perplexity", "val_loss", "val_perplexity"])

        if dls is not None:
            self.train_dl, self.val_dl = dls
            self.scaler = GradScaler()
            self.optim = torch.optim.Adam(self.model.parameters(), lr=train_config.lr / 25.0)
            self.sched = torch.optim.lr_scheduler.OneCycleLR(
                self.optim,
                max_lr=train_config.lr,
                epochs=train_config.epochs,
                steps_per_epoch=len(self.train_dl),
            )

    @property
    def checkpoint_path(self):
        return self.train_config.model_path / CHECKPOINT_NAME

    def save_model(self):
        self.train_config.model_path.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), self.checkpoint_path)

    def load_best_model(self):
        self.model.load_state_dict(torch.load(self.checkpoint_path, map_location=self.device))

    def train_one_epoch(self, epoch):
        prog = tqdm(self.train_dl, total=len(self.train_dl))
        running_loss = 0.0
        for image, input_ids, labels in prog:
            with autocast():
                image = image.to(self.device)
                input_ids = input_ids.to(self.device)
                labels = labels.to(self.device)

                loss = self.model(image, input_ids, labels)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optim)
                self.scaler.update()
                self.sched.step()
                self.optim.zero_grad(set_to_none=True)

                running_loss += loss.item()
                prog.set_description(f"train loss: {loss.item():.3f}")
            del image, input_ids, labels, loss

        train_loss = running_loss / len(self.train_dl)
        self.metrics.loc[epoch, ["train_loss", "train_perplexity"]] = (train_loss, np.exp(train_loss))

    @torch.no_grad()
    def valid_one_epoch(self, epoch):
        prog = tqdm(self.val_dl, total=len(self.val_dl))
        running_loss = 0.0
        for image, input_ids, labels in prog:
            with autocast():
                image = image.to(self.device)
                input_ids = input_ids.to(self.device)
                labels = labels.to(self.device)

                loss = self.model(image, input_ids, labels)
                running_loss += loss.item()
                prog.set_description(f"valid loss: {loss.item():.3f}")
            del image, input_ids, labels, loss

        val_loss = running_loss / len(self.val_dl)
        val_pxp = np.exp(val_loss)
        self.metrics.loc[epoch, ["val_loss", "val_perplexity"]] = (val_loss, val_pxp)
        return val_pxp

    @staticmethod
    def clean():
        gc.collect()
        torch.cuda.empty_cache()

    def fit(self):
        """Train for all epochs, saving the checkpoint with the lowest validation perplexity."""
        best_pxp = 1e9
        best_epoch = -1
        prog = tqdm(range(self.train_config.epochs))

        for epoch in prog:
            if epoch == self.train_config.freeze_epochs_gpt:
                self.model.unfreeze_gpt_layers()
                print("unfreezing GPT2 entirely...")
            if epoch == self.train_config.freeze_epochs_all:
                self.model.pretrained_layers_trainable(trainable=True)

            self.model.train()
            prog.set_description("training")
            self.train_one_epoch(epoch)
            self.clean()

            self.model.eval()
            prog.set_description("validating")
            pxp = self.valid_one_epoch(epoch)
            self.clean()

            print(self.metrics.tail(1))

            if pxp < best_pxp:
                best_pxp = pxp
                best_epoch = epoch
                print("saving best model...")
                self.save_model()

        return {"best_perplexity": best_pxp, "best_epoch": best_epoch}

    # --- Inference ---

    @torch.no_grad()
    def generate_caption(self, image_path, max_tokens=50, temperature=1.0, deterministic=False):
        """Caption a single image file."""
        self.model.eval()
        image = np.array(Image.open(image_path).convert("RGB"))
        image = self.gen_tfms(image=image)["image"].unsqueeze(0).to(self.device)
        sequence = torch.ones(1, 1).to(device=self.device).long() * self.tokenizer.bos_token_id

        caption = self.model.generate(
            image,
            sequence,
            eos_token_id=self.tokenizer.eos_token_id,
            max_tokens=max_tokens,
            temperature=temperature,
            deterministic=deterministic,
        )
        return self.tokenizer.decode(caption.numpy(), skip_special_tokens=True)


def save_training_plots(metrics, output_dir):
    """Save loss and perplexity curves."""
    plt.figure()
    plt.plot(metrics["train_loss"], color="red", label="train loss")
    plt.plot(metrics["val_loss"], color="orange", label="valid loss")
    plt.title("loss, lower=better")
    plt.legend()
    plt.savefig(output_dir / "loss.png", bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(metrics["train_perplexity"], color="blue", label="train perplexity")
    plt.plot(metrics["val_perplexity"], color="lightblue", label="valid perplexity")
    plt.title("perplexity, lower=better")
    plt.legend()
    plt.savefig(output_dir / "perplexity.png", bbox_inches="tight")
    plt.close()


def save_samples(trainer, val_df, output_dir, n_samples=5):
    """Caption random validation images with a random temperature and save the figures."""
    for i in range(n_samples):
        test_img, test_caption = val_df.sample(n=1).values[0][:2]
        t = np.random.uniform(0.5, 1.5)
        gen_caption = trainer.generate_caption(test_img, temperature=t, deterministic=False)
        plt.figure()
        plt.imshow(Image.open(test_img).convert("RGB"))
        plt.title(f"actual: {test_caption}\nmodel: {gen_caption}\ntemp: {t} deterministic generation: False")
        plt.axis("off")
        plt.savefig(output_dir / f"sample_{i}.png", bbox_inches="tight")
        plt.close()


def list_test_images(test_dir):
    """All .jpg/.png files in the test folder (os.listdir order, as in the notebook)."""
    paths = [os.path.join(test_dir, name) for name in os.listdir(test_dir)]
    return [p for p in paths if p.endswith(TEST_EXTENSIONS)]


def predict_test_set(trainer, test_dir, output_csv):
    """Greedy-decode a caption for each test image and write ID,Caption rows."""
    rows = []
    for path in tqdm(list_test_images(test_dir)):
        caption = trainer.generate_caption(path, deterministic=True)
        rows.append((Path(path).stem, caption))
    pd.DataFrame(rows, columns=["ID", "Caption"]).to_csv(output_csv, index=False)
    print(f"saved {len(rows)} captions to {output_csv}")


# --- Main ---

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["train", "sample", "predict"], default="train")
    parser.add_argument("--train-csv", default="train_captions.csv")
    parser.add_argument("--valid-csv", default="valid_captions.csv")
    parser.add_argument("--train-images", default="../Training/", help="folder with training <ID>.jpg images")
    parser.add_argument("--valid-images", default="../Valid/", help="folder with validation <ID>.jpg images")
    parser.add_argument("--test-images", default="../Test/", help="folder with test images (predict mode)")
    parser.add_argument("--split-source", choices=["valid", "train"], default="valid",
                        help="CSV that is split 90/10 into train/val (the notebook used the validation CSV)")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="checkpoint, metrics and plots go here")
    parser.add_argument("--predictions-csv", default="visionGpt2_gen_results.csv",
                        help="file name (inside --output-dir) for predict mode")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--freeze-epochs-gpt", type=int, default=DEFAULT_FREEZE_EPOCHS_GPT,
                        help="epoch at which the GPT-2 sub-layers are unfrozen")
    parser.add_argument("--freeze-epochs-all", type=int, default=DEFAULT_FREEZE_EPOCHS_ALL,
                        help="epoch at which all pretrained layers are unfrozen")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR, help="OneCycle max learning rate")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_config = SimpleNamespace(
        epochs=args.epochs,
        freeze_epochs_gpt=args.freeze_epochs_gpt,
        freeze_epochs_all=args.freeze_epochs_all,
        lr=args.lr,
        device=args.device,
        model_path=output_dir,
        batch_size=args.batch_size,
    )

    if args.mode == "predict":
        trainer = Trainer(MODEL_CONFIG, train_config)
        trainer.load_best_model()
        predict_test_set(trainer, args.test_images, output_dir / args.predictions_csv)
        return

    if args.split_source == "valid":
        source_df = load_caption_df(args.valid_csv, args.valid_images)
    else:
        source_df = load_caption_df(args.train_csv, args.train_images)
    train_df, val_df = train_test_split(source_df, test_size=0.1)
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    print(f"train: {len(train_df)}  val: {len(val_df)}")

    if args.mode == "sample":
        trainer = Trainer(MODEL_CONFIG, train_config)
        trainer.load_best_model()
        save_samples(trainer, val_df, output_dir)
        return

    tokenizer = load_tokenizer()
    dls = make_dataloaders(train_df, val_df, tokenizer, args.batch_size)
    trainer = Trainer(MODEL_CONFIG, train_config, dls)
    result = trainer.fit()
    print(result)

    trainer.metrics.to_csv(output_dir / "metrics.csv")
    save_training_plots(trainer.metrics, output_dir)
    trainer.load_best_model()
    save_samples(trainer, val_df, output_dir)


if __name__ == "__main__":
    main()
