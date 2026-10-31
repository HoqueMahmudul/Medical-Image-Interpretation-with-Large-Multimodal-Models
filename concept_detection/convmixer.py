"""ConvMixer-256/8 for multi-label UMLS concept detection (ImageCLEFmedical 2024, ROCOv2).

ConvMixer (patch-embedding conv stem followed by depthwise/pointwise conv blocks)
trained from scratch with a sigmoid output per concept (CUI) and binary
cross-entropy. A CUI is predicted when its probability exceeds 0.5.

Modes:
    train     Train on the training concepts, keep the checkpoint with the best
              validation accuracy, save the accuracy curve and filter visualisations.
    evaluate  Report loss/accuracy/precision/recall/AUC of the saved model on the validation set.
    predict   Predict CUIs for every file in the test folder and write the
              ImageCLEF submission file (``<ID>|<CUI>;<CUI>...``, no header).

Data layout:
    train_concepts.csv / valid_concepts.csv with columns ``ID`` and ``CUIs``
    (CUIs separated by ';'); images stored as ``<images-dir>/<ID>.jpg``.
    The label set is built from the training CSV, so evaluate/predict need it too.

Usage:
    python convmixer.py --mode train --train-csv ../train_concepts.csv --valid-csv ../valid_concepts.csv
    python convmixer.py --mode predict --test-images ../Test

Requirements:
    tensorflow 2.x (Keras), numpy, pandas, scikit-learn, matplotlib
"""

import argparse
import os
import warnings
from pathlib import Path

import keras
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers
from sklearn.preprocessing import MultiLabelBinarizer

# --- Defaults ---

LEARNING_RATE = 0.001
NUM_EPOCHS = 2
BATCH_SIZE = 8
IMAGE_SIZE = 192
FILTERS = 256
DEPTH = 8
KERNEL_SIZE = 5
PATCH_SIZE = 2
THRESHOLD = 0.5
MAX_LABELS_PER_IMAGE = 100
CHECKPOINT_NAME = "checkpoint.keras"
SUBMISSION_NAME = "taskA_N2RjY_run2.csv"
PATCH_EMBEDDING_LAYER = 2  # index of the stem Conv2D in model.layers
MID_DEPTHWISE_LAYER = 26  # a depthwise conv from the middle of the network


# --- Data ---

def load_concepts(csv_path, images_dir):
    """Return image paths and CUI lists from a concepts CSV."""
    df = pd.read_csv(csv_path)
    paths = [images_dir + str(image_id) + ".jpg" for image_id in df["ID"]]
    cuis = [cui_string.split(";") for cui_string in df["CUIs"]]
    return paths, cuis


def build_binarizer(train_cuis):
    """MultiLabelBinarizer over training CUIs, in order of first appearance."""
    unique_cui = list(dict.fromkeys(cui for cuis in train_cuis for cui in cuis))
    mlb = MultiLabelBinarizer(classes=unique_cui)
    mlb.fit(train_cuis)
    return mlb


def make_preprocess(image_size):
    """Image loader with random flip / 90-degree rotation (applied to every split, as in the notebook)."""

    def preprocess_image(image_path):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.random_flip_left_right(image)
        image = tf.image.rot90(image, k=tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
        image = tf.image.resize(image, (image_size, image_size))
        return tf.cast(image, tf.float32) / 255.0

    return preprocess_image


def make_image_dataset(paths, preprocess, batch_size):
    ds = tf.data.Dataset.from_tensor_slices(paths)
    ds = ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size, num_parallel_calls=tf.data.AUTOTUNE)


def make_labeled_dataset(paths, labels, preprocess, batch_size):
    images = make_image_dataset(paths, preprocess, batch_size)
    targets = tf.data.Dataset.from_tensor_slices(labels).batch(batch_size, num_parallel_calls=tf.data.AUTOTUNE)
    return tf.data.Dataset.zip((images, targets))


# --- Model ---

def activation_block(x):
    x = layers.Activation("gelu")(x)
    return layers.BatchNormalization()(x)


def conv_stem(x, filters: int, patch_size: int):
    x = layers.Conv2D(filters, kernel_size=patch_size, strides=patch_size)(x)
    return activation_block(x)


def conv_mixer_block(x, filters: int, kernel_size: int):
    x0 = x
    x = layers.DepthwiseConv2D(kernel_size=kernel_size, padding="same")(x)
    x = layers.Add()([activation_block(x), x0])
    x = layers.Conv2D(filters, kernel_size=1)(x)
    return activation_block(x)


def get_conv_mixer_256_8(num_classes, image_size=IMAGE_SIZE, filters=FILTERS, depth=DEPTH,
                         kernel_size=KERNEL_SIZE, patch_size=PATCH_SIZE):
    """ConvMixer with a sigmoid multi-label head."""
    inputs = keras.Input((image_size, image_size, 3))
    # Note: inputs are already scaled to [0, 1] by the loader, so this rescales a second time.
    x = layers.Rescaling(scale=1.0 / 255)(inputs)
    x = conv_stem(x, filters, patch_size)
    for _ in range(depth):
        x = conv_mixer_block(x, filters, kernel_size)
    x = layers.GlobalAvgPool2D()(x)
    outputs = layers.Dense(num_classes, activation="sigmoid")(x)
    return keras.Model(inputs, outputs)


# --- Training ---

def compile_model(model, learning_rate):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall(),
            tf.keras.metrics.AUC(),
        ],
    )


def run_experiment(model, train_ds, valid_ds, epochs, learning_rate, checkpoint_path):
    """Fit with best-val-accuracy checkpointing, then reload the best weights."""
    compile_model(model, learning_rate)
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        str(checkpoint_path),
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=False,
    )
    history = model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=epochs,
        callbacks=[checkpoint_callback],
    )
    model.load_weights(str(checkpoint_path))
    return history, model


def save_accuracy_plot(history, path):
    plt.figure()
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def save_filter_plot(weights, path, num_filters=FILTERS):
    """Min-max normalise conv weights and plot each of the 256 filters in a 16x16 grid."""
    p_min, p_max = weights.min(), weights.max()
    weights = (weights - p_min) / (p_max - p_min)

    plt.figure(figsize=(8, 8))
    for i in range(num_filters):
        current_weight = weights[:, :, :, i]
        if current_weight.shape[-1] == 1:
            current_weight = current_weight.squeeze()
        ax = plt.subplot(16, 16, i + 1)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.imshow(current_weight)
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def save_filter_visualisations(model, output_dir):
    """Learned patch embeddings and one mid-network depthwise kernel."""
    patch_embeddings = model.layers[PATCH_EMBEDDING_LAYER].get_weights()[0]
    save_filter_plot(patch_embeddings, output_dir / "patch_embeddings.png")

    kernel = model.layers[MID_DEPTHWISE_LAYER].get_weights()[0]
    kernel = np.expand_dims(kernel.squeeze(), axis=2)
    save_filter_plot(kernel, output_dir / f"depthwise_kernel_layer{MID_DEPTHWISE_LAYER}.png")


# --- Inference ---

def predict_test_set(model, mlb, test_dir, preprocess, batch_size, out_csv):
    """Threshold sigmoid outputs and write ``<ID>|<CUI;CUI;...>`` lines."""
    test_paths = [os.path.join(test_dir, name) for name in os.listdir(test_dir)]
    test_ds = make_image_dataset(test_paths, preprocess, batch_size).prefetch(tf.data.AUTOTUNE)

    print("[INFO] generating predictions...")
    y_pred = model.predict(test_ds, verbose=1)
    y_pred_labels = mlb.inverse_transform((y_pred > THRESHOLD).astype(int))

    print("[INFO] saving predictions...")
    rows = []
    for path, labels in zip(test_paths, y_pred_labels):
        unique_labels = list(dict.fromkeys(labels))[:MAX_LABELS_PER_IMAGE]
        filename = os.path.splitext(os.path.basename(path))[0]
        rows.append([filename + "|" + ";".join(unique_labels)])
    pd.DataFrame(rows).to_csv(out_csv, index=False, header=False)
    print(f"saved {len(rows)} predictions to {out_csv}")


# --- Main ---

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["train", "evaluate", "predict"], default="train")
    parser.add_argument("--train-csv", default="../train_concepts.csv")
    parser.add_argument("--valid-csv", default="../valid_concepts.csv")
    parser.add_argument("--train-images", default="../Training/")
    parser.add_argument("--valid-images", default="../Valid/")
    parser.add_argument("--test-images", default="../Test")
    parser.add_argument("--output-dir", default="convmixer_output", help="checkpoint, plots and submission go here")
    parser.add_argument("--submission-csv", default=SUBMISSION_NAME, help="file name inside --output-dir")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / CHECKPOINT_NAME

    warnings.filterwarnings("ignore")
    tf.keras.backend.clear_session()

    train_paths, train_cuis = load_concepts(args.train_csv, args.train_images)
    mlb = build_binarizer(train_cuis)
    preprocess = make_preprocess(args.image_size)

    if args.mode == "predict":
        model = keras.models.load_model(str(checkpoint_path))
        predict_test_set(model, mlb, args.test_images, preprocess, args.batch_size,
                         output_dir / args.submission_csv)
        return

    valid_paths, valid_cuis = load_concepts(args.valid_csv, args.valid_images)
    valid_ds = make_labeled_dataset(valid_paths, mlb.transform(valid_cuis), preprocess, args.batch_size)

    if args.mode == "evaluate":
        model = keras.models.load_model(str(checkpoint_path))
        results = model.evaluate(valid_ds, verbose=1, return_dict=True)
        for name, value in results.items():
            print(f"Validation {name}: {value}")
        return

    train_ds = make_labeled_dataset(train_paths, mlb.transform(train_cuis), preprocess, args.batch_size)
    model = get_conv_mixer_256_8(num_classes=len(mlb.classes_), image_size=args.image_size)
    history, model = run_experiment(model, train_ds, valid_ds, args.epochs, args.lr, checkpoint_path)
    save_accuracy_plot(history, output_dir / "accuracy.png")
    save_filter_visualisations(model, output_dir)


if __name__ == "__main__":
    main()
