"""CNN-Transformer caption model: DenseNet121 (CheXNet weights) encoder + Transformer decoder.

Image features come from a DenseNet121 initialised with CheXNet weights; its last
``relu`` feature map goes through a 1x1 Conv2D(2048) and is reshaped to 49 tokens.
A one-block Transformer encoder (1 head) and a one-block Transformer decoder
(2 heads, learned positional embeddings) are trained on top with teacher forcing.
The CNN itself is not updated: only encoder and decoder variables get gradients.
Captions are generated with greedy search.

Modes:
    train     Pre-process captions, split the pooled train+valid images 80/15/5,
              train the model, save weights, vocabulary, split pickles and loss/acc plots.
    predict   Caption every .jpg/.png in the test folder and write a CSV.
    evaluate  Caption the held-out 5% split and report BLEU-1..4, ROUGE and METEOR.

Data layout (ImageCLEFmedical 2024 caption task, ROCOv2):
    train_captions.csv / valid_captions.csv with columns ``ID`` and ``Caption``;
    images stored as ``<images-dir>/<ID>.jpg``.
    CheXNet Keras weights: ``brucechou1983_CheXNet_Keras_0.3.0_weights.h5``.

Usage:
    python cnn_transformer.py --mode train --chexnet-weights brucechou1983_CheXNet_Keras_0.3.0_weights.h5
    python cnn_transformer.py --mode predict --test-images ../Test/
    python cnn_transformer.py --mode evaluate

Requirements:
    tensorflow 2.x with Keras 2 (uses tf.keras.losses.Reduction and keras.layers.TextVectorization),
    numpy, pandas, opencv-python, nltk, rouge, matplotlib, tqdm
"""

import argparse
import json
import os
import pickle
import random
import re
import warnings
from collections import Counter
from pathlib import Path

import cv2
import keras
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers
from keras.layers import TextVectorization
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge import Rouge
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import Conv2D, Dense, GlobalAveragePooling2D, Reshape
from tqdm import tqdm

# --- Defaults ---

SEED = 0
IMAGE_SIZE = (224, 224)
VOCAB_SIZE = 33227  # fixed in the notebook (len(vocab_list) of its data)
EMBED_DIM = 512  # image and token embedding size
FF_DIM = 512  # feed-forward units per layer
ENCODER_HEADS = 1
DECODER_HEADS = 2
BATCH_SIZE = 128
EPOCHS = 200
SCHEDULE_EPOCHS = 50  # the warm-up schedule was sized for 50 epochs, although fit ran 200
LEARNING_RATE = 1e-4
WARMUP_FRACTION_DIVISOR = 15  # warm-up steps = total schedule steps // 15
CAPTION_LENGTH_COVERAGE = 0.99
TRAIN_FRACTION = 0.8
VAL_FRACTION = 0.15
TEST_EXTENSIONS = ("jpg", "png")

CHEXNET_WEIGHTS = "brucechou1983_CheXNet_Keras_0.3.0_weights.h5"
CHECKPOINT_PREFIX = "caption_model_ckpt"
CONFIG_FILE = "caption_config.json"
PICKLE_DIR = "pickles_clef24"

# Punctuation removed by the TextVectorization standardiser; < and > are kept for <start>/<end>.
STRIP_CHARS = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~".replace("<", "").replace(">", "")


# --- Setup ---

def configure_gpus():
    """Enable memory growth on every visible GPU."""
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        print("No GPU is available.")
        return
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices("GPU")
        print(f"{len(gpus)} Physical GPU(s), {len(logical_gpus)} Logical GPU(s) configured.")
    except RuntimeError as e:
        print(f"RuntimeError in configuring GPUs: {e}")


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"


# --- Data ---

class CaptionDataProcessor:
    """Load caption CSVs, clean captions and split images into train/val/test."""

    def __init__(self, train_csv, valid_csv, train_images_path, valid_images_path):
        self.train_csv = train_csv
        self.valid_csv = valid_csv
        self.train_images_path = train_images_path
        self.valid_images_path = valid_images_path
        self.image_caption_pairs = {}
        self.caption_ls = []
        self.vocab_list = set()
        self.cap_len = []

    def load_data(self):
        df_train = pd.read_csv(self.train_csv, index_col=False)
        df_valid = pd.read_csv(self.valid_csv, index_col=False)
        self.process_data(df_train, self.train_images_path, "Training Image and Caption Data")
        self.process_data(df_valid, self.valid_images_path, "Validation Image and Caption Data")

    @staticmethod
    def process_caption(text):
        """Lowercase, drop punctuation and any word containing a digit, add <START>/<END>."""
        caption = text.lower().strip()
        caption = re.sub(r"-", " ", caption)
        caption = re.sub(r"[^a-zA-Z0-9\s]", "", caption)
        words = [w for w in caption.split() if not any(ch.isdigit() for ch in w)]
        return "<START> " + " ".join(words) + " <END>"

    def process_data(self, df, image_path_prefix, desc):
        for _, row in tqdm(df.iterrows(), total=df.shape[0], desc=desc):
            img_path = os.path.join(image_path_prefix, str(row["ID"]) + ".jpg")
            caption = self.process_caption(row["Caption"])
            if not os.path.exists(img_path):
                print(f"File not found: {img_path}")
                continue
            self.image_caption_pairs.setdefault(img_path, []).append(caption)
            self.caption_ls.append(caption)
            words = caption.split(" ")
            self.vocab_list.update(words)
            self.cap_len.append(len(words))

    def train_val_test_split(self, train_size=TRAIN_FRACTION, val_size=VAL_FRACTION, shuffle=True):
        all_images = list(self.image_caption_pairs.keys())
        if shuffle:
            np.random.shuffle(all_images)

        total_size = len(self.image_caption_pairs)
        train_end = int(total_size * train_size)
        val_end = train_end + int(total_size * val_size)

        self.training_data = {k: self.image_caption_pairs[k] for k in all_images[:train_end]}
        self.validation_data = {k: self.image_caption_pairs[k] for k in all_images[train_end:val_end]}
        self.test_data = {k: self.image_caption_pairs[k] for k in all_images[val_end:]}
        return self.training_data, self.validation_data, self.test_data

    def save_datasets(self, save_dir):
        """Pickle the full dataset and the three splits."""
        os.makedirs(save_dir, exist_ok=True)
        for name, data in [
            ("main_dataset.pkl", self.image_caption_pairs),
            ("train.pkl", self.training_data),
            ("valid.pkl", self.validation_data),
            ("test.pkl", self.test_data),
        ]:
            with open(os.path.join(save_dir, name), "wb") as f:
                pickle.dump(data, f)
        print("Datasets saved successfully.")


def caption_length_at_coverage(cap_len, coverage=CAPTION_LENGTH_COVERAGE):
    """Return the caption length used as SEQ_LENGTH.

    Lengths are visited from most to least frequent and the length at which the
    cumulative count first reaches `coverage` of all captions is returned
    (this is the notebook's rule, not a length percentile).
    """
    frequency = Counter(cap_len)
    total_items = sum(frequency.values())
    cumulative = 0
    for num, freq in sorted(frequency.items(), key=lambda x: x[1], reverse=True):
        cumulative += freq
        if cumulative >= total_items * coverage:
            return num
    return num


def custom_standardization(input_string):
    lowercase = tf.strings.lower(input_string)
    return tf.strings.regex_replace(lowercase, "[%s]" % re.escape(STRIP_CHARS), "")


def build_vectorization(seq_length, vocab_size=VOCAB_SIZE):
    return TextVectorization(
        max_tokens=vocab_size,
        output_mode="int",
        output_sequence_length=seq_length,
        standardize=custom_standardization,
    )


class DatasetBuilder:
    """tf.data pipeline: decode + resize + scale image, vectorise caption."""

    def __init__(self, vectorization_layer, batch_size, img_size, autotune=tf.data.AUTOTUNE):
        self.vectorization_layer = vectorization_layer
        self.batch_size = batch_size
        self.img_size = img_size
        self.autotune = autotune

    def decode_and_resize(self, img_path):
        img = tf.io.read_file(img_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, self.img_size)
        return tf.cast(img, tf.float32) / 255.0

    def process_input(self, img_path, caption):
        img = self.decode_and_resize(img_path)
        caption = self.vectorization_layer(tf.expand_dims(caption, 0))
        return img, tf.squeeze(caption, 0)

    def make_dataset(self, image_paths, captions):
        dataset = tf.data.Dataset.from_tensor_slices((image_paths, captions))
        dataset = dataset.shuffle(buffer_size=len(image_paths))
        dataset = dataset.map(self.process_input, num_parallel_calls=self.autotune)
        return dataset.batch(self.batch_size).prefetch(self.autotune)


# --- Model ---

def get_cnn_model(image_size=IMAGE_SIZE, chexnet_weights=None):
    """DenseNet121 feature extractor -> 1x1 Conv2D(2048) -> 49 image tokens.

    The 14-way CheXNet head is only rebuilt so that the CheXNet weight file loads;
    features are taken from the final 'relu' layer.
    """
    base_model = DenseNet121(include_top=False, input_shape=(*image_size, 3))
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(14, activation="sigmoid", name="chexnet_output")(x)
    chexnet = tf.keras.Model(inputs=base_model.input, outputs=x)
    if chexnet_weights:
        chexnet.load_weights(chexnet_weights)
    x = chexnet.get_layer("relu").output
    x = Conv2D(2048, (1, 1), padding="same", activation="relu")(x)
    x = Reshape((49, 2048))(x)
    return tf.keras.Model(inputs=base_model.input, outputs=x)


class TransformerEncoderBlock(layers.Layer):
    """LayerNorm -> Dense projection -> self-attention with residual + LayerNorm."""

    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention_1 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim, dropout=0.0)
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.dense_1 = layers.Dense(embed_dim, activation="relu")

    def call(self, inputs, training, mask=None):
        inputs = self.dense_1(self.layernorm_1(inputs))
        attention_output_1 = self.attention_1(
            query=inputs, value=inputs, key=inputs, attention_mask=None, training=training
        )
        return self.layernorm_2(inputs + attention_output_1)


class PositionalEmbedding(layers.Layer):
    """Scaled token embedding plus learned position embedding; 0 is the padding id."""

    def __init__(self, sequence_length, vocab_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.token_embeddings = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.position_embeddings = layers.Embedding(input_dim=sequence_length, output_dim=embed_dim)
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.embed_scale = tf.math.sqrt(tf.cast(embed_dim, tf.float32))

    def call(self, inputs):
        length = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_tokens = self.token_embeddings(inputs) * self.embed_scale
        return embedded_tokens + self.position_embeddings(positions)

    def compute_mask(self, inputs, mask=None):
        return tf.math.not_equal(inputs, 0)


class TransformerDecoderBlock(layers.Layer):
    """Causal self-attention, cross-attention to image tokens, FFN, softmax over vocab."""

    def __init__(self, embed_dim, ff_dim, num_heads, seq_length, vocab_size, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim
        self.num_heads = num_heads
        self.attention_1 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim, dropout=0.1)
        self.attention_2 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim, dropout=0.1)
        self.ffn_layer_1 = layers.Dense(ff_dim, activation="relu")
        self.ffn_layer_2 = layers.Dense(embed_dim)

        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.layernorm_3 = layers.LayerNormalization()

        self.embedding = PositionalEmbedding(
            embed_dim=embed_dim, sequence_length=seq_length, vocab_size=vocab_size
        )
        self.out = layers.Dense(vocab_size, activation="softmax")

        self.dropout_1 = layers.Dropout(0.3)
        self.dropout_2 = layers.Dropout(0.5)
        self.supports_masking = True

    def call(self, inputs, encoder_outputs, training, mask=None):
        inputs = self.embedding(inputs)
        causal_mask = self.get_causal_attention_mask(inputs)

        # A mask is always passed by the callers; the masks are only defined in that case.
        if mask is not None:
            padding_mask = tf.cast(mask[:, :, tf.newaxis], dtype=tf.int32)
            combined_mask = tf.cast(mask[:, tf.newaxis, :], dtype=tf.int32)
            combined_mask = tf.minimum(combined_mask, causal_mask)

        attention_output_1 = self.attention_1(
            query=inputs, value=inputs, key=inputs, attention_mask=combined_mask, training=training
        )
        out_1 = self.layernorm_1(inputs + attention_output_1)

        attention_output_2 = self.attention_2(
            query=out_1, value=encoder_outputs, key=encoder_outputs,
            attention_mask=padding_mask, training=training,
        )
        out_2 = self.layernorm_2(out_1 + attention_output_2)

        ffn_out = self.ffn_layer_1(out_2)
        ffn_out = self.dropout_1(ffn_out, training=training)
        ffn_out = self.ffn_layer_2(ffn_out)
        ffn_out = self.layernorm_3(ffn_out + out_2, training=training)
        ffn_out = self.dropout_2(ffn_out, training=training)
        return self.out(ffn_out)

    def get_causal_attention_mask(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size, sequence_length = input_shape[0], input_shape[1]
        i = tf.range(sequence_length)[:, tf.newaxis]
        j = tf.range(sequence_length)
        mask = tf.cast(i >= j, dtype="int32")
        mask = tf.reshape(mask, (1, input_shape[1], input_shape[1]))
        mult = tf.concat([tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)], axis=0)
        return tf.tile(mask, mult)


class ImageCaptioningModel(keras.Model):
    """Wraps CNN, encoder and decoder with custom train/test steps (masked loss and accuracy)."""

    def __init__(self, cnn_model, encoder, decoder, image_aug=None):
        super().__init__()
        self.cnn_model = cnn_model
        self.encoder = encoder
        self.decoder = decoder
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.acc_tracker = keras.metrics.Mean(name="accuracy")
        self.image_aug = image_aug

    def calculate_loss(self, y_true, y_pred, mask):
        loss = self.loss(y_true, y_pred)
        mask = tf.cast(mask, dtype=loss.dtype)
        loss *= mask
        return tf.reduce_sum(loss) / tf.reduce_sum(mask)

    def calculate_accuracy(self, y_true, y_pred, mask):
        accuracy = tf.equal(y_true, tf.argmax(y_pred, axis=2))
        accuracy = tf.math.logical_and(mask, accuracy)
        accuracy = tf.cast(accuracy, dtype=tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)
        return tf.reduce_sum(accuracy) / tf.reduce_sum(mask)

    def _compute_caption_loss_and_acc(self, img_embed, batch_seq, training=True):
        encoder_out = self.encoder(img_embed, training=training)
        batch_seq_inp = batch_seq[:, :-1]
        batch_seq_true = batch_seq[:, 1:]
        mask = tf.math.not_equal(batch_seq_true, 0)
        batch_seq_pred = self.decoder(batch_seq_inp, encoder_out, training=training, mask=mask)
        loss = self.calculate_loss(batch_seq_true, batch_seq_pred, mask)
        acc = self.calculate_accuracy(batch_seq_true, batch_seq_pred, mask)
        return loss, acc

    def train_step(self, batch_data):
        batch_img, batch_seq = batch_data
        if self.image_aug:
            batch_img = self.image_aug(batch_img)
        img_embed = self.cnn_model(batch_img)

        with tf.GradientTape() as tape:
            loss, acc = self._compute_caption_loss_and_acc(img_embed, batch_seq, training=True)
            self.loss_tracker.update_state(loss)
            self.acc_tracker.update_state(acc)

        # The CNN is frozen implicitly: only encoder/decoder variables are updated.
        train_vars = self.encoder.trainable_variables + self.decoder.trainable_variables
        grads = tape.gradient(loss, train_vars)
        self.optimizer.apply_gradients(zip(grads, train_vars))
        return {"loss": self.loss_tracker.result(), "acc": self.acc_tracker.result()}

    def test_step(self, batch_data):
        batch_img, batch_seq = batch_data
        img_embed = self.cnn_model(batch_img)
        loss, acc = self._compute_caption_loss_and_acc(img_embed, batch_seq, training=False)
        self.loss_tracker.update_state(loss)
        self.acc_tracker.update_state(acc)
        return {"loss": self.loss_tracker.result(), "acc": self.acc_tracker.result()}

    @property
    def metrics(self):
        # Listed so that Keras resets them at the start of each epoch.
        return [self.loss_tracker, self.acc_tracker]


def build_caption_model(seq_length, chexnet_weights=None):
    cnn_model = get_cnn_model(IMAGE_SIZE, chexnet_weights)
    encoder = TransformerEncoderBlock(embed_dim=EMBED_DIM, dense_dim=FF_DIM, num_heads=ENCODER_HEADS)
    decoder = TransformerDecoderBlock(
        embed_dim=EMBED_DIM, ff_dim=FF_DIM, num_heads=DECODER_HEADS,
        seq_length=seq_length, vocab_size=VOCAB_SIZE,
    )
    return ImageCaptioningModel(cnn_model=cnn_model, encoder=encoder, decoder=decoder)


def model_checkpoint(caption_model):
    """TF checkpoint over the three sub-models (the wrapper model is never called directly)."""
    return tf.train.Checkpoint(
        cnn_model=caption_model.cnn_model,
        encoder=caption_model.encoder,
        decoder=caption_model.decoder,
    )


def build_and_restore(caption_model, seq_length, checkpoint_prefix):
    """Create all variables with a dummy forward pass, then restore trained weights."""
    dummy_img = tf.zeros((1, *IMAGE_SIZE, 3))
    encoded = caption_model.encoder(caption_model.cnn_model(dummy_img), training=False)
    dummy_seq = tf.zeros((1, seq_length - 1), dtype=tf.int64)
    caption_model.decoder(dummy_seq, encoded, training=False, mask=tf.ones_like(dummy_seq, dtype=tf.bool))
    model_checkpoint(caption_model).read(checkpoint_prefix).expect_partial()


# --- Training ---

class LRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Linear warm-up to a constant learning rate."""

    def __init__(self, post_warmup_learning_rate, warmup_steps):
        super().__init__()
        self.post_warmup_learning_rate = post_warmup_learning_rate
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        global_step = tf.cast(step, tf.float32)
        warmup_steps = tf.cast(self.warmup_steps, tf.float32)
        warmup_learning_rate = self.post_warmup_learning_rate * (global_step / warmup_steps)
        return tf.cond(
            global_step < warmup_steps,
            lambda: warmup_learning_rate,
            lambda: self.post_warmup_learning_rate,
        )


def save_history_plot(history, path):
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    acc = history.history["acc"]
    val_acc = history.history["val_acc"]
    epochs = range(1, len(loss) + 1)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, "b-", label="Training Loss")
    plt.plot(epochs, val_loss, "r-", label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, acc, "b-", label="Training Accuracy")
    plt.plot(epochs, val_acc, "r-", label="Validation Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.savefig(path, bbox_inches="tight")
    plt.close()


def train(args, output_dir):
    """Prepare data, train the model and save everything needed for inference."""
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        processor = CaptionDataProcessor(args.train_csv, args.valid_csv, args.train_images, args.valid_images)
        processor.load_data()
        train_data, valid_data, test_data = processor.train_val_test_split()
        processor.save_datasets(output_dir / PICKLE_DIR)
        print("Number of records in the main dataset: ", len(processor.image_caption_pairs))
        print("Number of training samples: ", len(train_data))
        print("Number of validation samples: ", len(valid_data))
        print("Number of test samples: ", len(test_data))

        seq_length = caption_length_at_coverage(processor.cap_len)
        print(f"SEQ_LENGTH (caption length at {CAPTION_LENGTH_COVERAGE:.0%} coverage): {seq_length}")

        vectorization = build_vectorization(seq_length)
        vectorization.adapt(processor.caption_ls)

        builder = DatasetBuilder(vectorization, args.batch_size, IMAGE_SIZE)
        train_dataset = builder.make_dataset(list(train_data.keys()), list(train_data.values()))
        valid_dataset = builder.make_dataset(list(valid_data.keys()), list(valid_data.values()))

        caption_model = build_caption_model(seq_length, args.chexnet_weights)

        cross_entropy = keras.losses.SparseCategoricalCrossentropy(
            from_logits=False, reduction=tf.keras.losses.Reduction.NONE
        )
        num_train_steps = len(train_dataset) * args.schedule_epochs
        num_warmup_steps = num_train_steps // WARMUP_FRACTION_DIVISOR
        lr_schedule = LRSchedule(post_warmup_learning_rate=args.lr, warmup_steps=num_warmup_steps)
        caption_model.compile(optimizer=tf.keras.optimizers.Adam(lr_schedule), loss=cross_entropy)

        history = caption_model.fit(train_dataset, epochs=args.epochs, validation_data=valid_dataset)

    model_checkpoint(caption_model).write(str(output_dir / CHECKPOINT_PREFIX))
    with open(output_dir / CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump({"seq_length": int(seq_length), "vocabulary": vectorization.get_vocabulary()}, f)
    save_history_plot(history, output_dir / "loss_accuracy.png")
    print(f"saved model and vocabulary to {output_dir}")


# --- Inference ---

def load_trained(output_dir):
    """Rebuild vectoriser and model from the files written by `train`."""
    with open(output_dir / CONFIG_FILE, encoding="utf-8") as f:
        config = json.load(f)
    seq_length = config["seq_length"]
    vectorization = build_vectorization(seq_length)
    vectorization.set_vocabulary(config["vocabulary"])

    caption_model = build_caption_model(seq_length)
    build_and_restore(caption_model, seq_length, str(output_dir / CHECKPOINT_PREFIX))
    return caption_model, vectorization, seq_length


def greedy_decode(caption_model, vectorization, image, seq_length):
    """Greedy search from '<start>' until '<end>' or SEQ_LENGTH - 1 tokens."""
    cnn_img = caption_model.cnn_model(image)
    encoded_img = caption_model.encoder(cnn_img, training=False)

    vocab = vectorization.get_vocabulary()
    index_lookup = dict(zip(range(len(vocab)), vocab))

    decoded_caption = "<start> "
    for i in range(seq_length - 1):
        tokenized_caption = vectorization([decoded_caption])[:, :-1]
        mask = tf.math.not_equal(tokenized_caption, 0)
        predictions = caption_model.decoder(tokenized_caption, encoded_img, training=False, mask=mask)
        sampled_token = index_lookup[np.argmax(predictions[0, i, :])]
        if sampled_token == "<end>":
            break
        decoded_caption += " " + sampled_token

    return decoded_caption.replace("<start> ", "").replace("<end>", "").strip()


def load_test_image(path):
    """Test-time loading used by the notebook: OpenCV (BGR, 0-255) + resize_with_pad."""
    img = tf.convert_to_tensor(cv2.imread(path))
    img = tf.image.resize_with_pad(img, target_height=IMAGE_SIZE[0], target_width=IMAGE_SIZE[1])
    return tf.expand_dims(img, 0)


def predict(args, output_dir):
    """Caption every test image and save ID,Caption rows."""
    caption_model, vectorization, seq_length = load_trained(output_dir)

    test_paths = [os.path.join(args.test_images, name) for name in sorted(os.listdir(args.test_images))]
    test_paths = [p for p in test_paths if p.endswith(TEST_EXTENSIONS)]

    rows = []
    for path in tqdm(test_paths, desc="Processing Test Dataset"):
        caption = greedy_decode(caption_model, vectorization, load_test_image(path), seq_length)
        rows.append((Path(path).stem, caption))

    out_csv = output_dir / args.predictions_csv
    pd.DataFrame(rows, columns=["ID", "Caption"]).to_csv(out_csv, index=False)
    print(f"saved {len(rows)} captions to {out_csv}")


# --- Evaluation ---

def mean_bleu(pred_ls, act_ls):
    """Mean sentence BLEU-1..4.

    Captions are passed to sentence_bleu as raw strings (as in the notebook), so
    n-grams are counted over characters, not words.
    """
    bleu1, bleu2, bleu3, bleu4 = [], [], [], []
    for predict_tokens, true_tokens in zip(pred_ls, act_ls):
        bleu1.append(sentence_bleu([true_tokens], predict_tokens, weights=(1, 0, 0, 0)))
        bleu2.append(sentence_bleu([true_tokens], predict_tokens, weights=(0.5, 0.5, 0, 0)))
        bleu3.append(sentence_bleu([true_tokens], predict_tokens, weights=(0.33, 0.33, 0.33, 0)))
        bleu4.append(sentence_bleu([true_tokens], predict_tokens, weights=(0.25, 0.25, 0.25, 0.25)))
    return np.mean(bleu1), np.mean(bleu2), np.mean(bleu3), np.mean(bleu4)


def mean_meteor(predictions, references):
    scores = [meteor_score([ref.split()], pred.split()) for ref, pred in zip(references, predictions)]
    return np.mean(scores)


def evaluate(args, output_dir):
    """Caption the held-out test split saved by `train` and print BLEU/ROUGE/METEOR."""
    caption_model, vectorization, seq_length = load_trained(output_dir)
    with open(output_dir / PICKLE_DIR / "test.pkl", "rb") as f:
        test_data = pickle.load(f)

    # Same image pre-processing as the training pipeline.
    image_loader = DatasetBuilder(vectorization, 1, IMAGE_SIZE)
    predicted, actual, ids = [], [], []
    for img_path, captions in tqdm(test_data.items(), desc="Evaluating"):
        image = tf.expand_dims(image_loader.decode_and_resize(img_path), 0)
        predicted.append(greedy_decode(caption_model, vectorization, image, seq_length))
        actual.append(captions[0].replace("<START>", "").replace("<END>", "").strip().lower())
        ids.append(Path(img_path).stem)

    pd.DataFrame({"ID": ids, "Caption": predicted, "Reference": actual}).to_csv(
        output_dir / "heldout_predictions.csv", index=False
    )

    bleu1, bleu2, bleu3, bleu4 = mean_bleu(predicted, actual)
    print("Bleu Score 1-gram: ", bleu1)
    print("Bleu Score 2-gram: ", bleu2)
    print("Bleu Score 3-gram: ", bleu3)
    print("Bleu Score 4-gram: ", bleu4)
    print(Rouge().get_scores(predicted, actual, avg=True))
    print(f"Mean METEOR score: {mean_meteor(predicted, actual)}")


# --- Main ---

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["train", "predict", "evaluate"], default="train")
    parser.add_argument("--train-csv", default="train_captions.csv")
    parser.add_argument("--valid-csv", default="valid_captions.csv")
    parser.add_argument("--train-images", default="../Training/")
    parser.add_argument("--valid-images", default="../Valid/")
    parser.add_argument("--test-images", default="../Test/")
    parser.add_argument("--chexnet-weights", default=CHEXNET_WEIGHTS, help="CheXNet Keras .h5 (train mode)")
    parser.add_argument("--output-dir", default="cnn_transformer_output")
    parser.add_argument("--predictions-csv", default="enc_dec_caption.csv",
                        help="file name (inside --output-dir) for predict mode")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--schedule-epochs", type=int, default=SCHEDULE_EPOCHS,
                        help="epoch count used to size the LR warm-up (notebook: 50)")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="post-warm-up learning rate")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tf.keras.backend.clear_session()
    warnings.filterwarnings("ignore")
    configure_gpus()
    seed_everything(SEED)

    if args.mode == "train":
        train(args, output_dir)
    elif args.mode == "predict":
        predict(args, output_dir)
    else:
        evaluate(args, output_dir)


if __name__ == "__main__":
    main()
