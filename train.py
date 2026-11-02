"""Train and compare CNNs for binary brain tumor detection (MRI, JPG).

"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,
                             confusion_matrix, roc_auc_score)
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers

SEED = 42
AUTOTUNE = tf.data.AUTOTUNE


# ----------------------------------------------------------------- data
def collect_files(data_dir: Path):
    paths, labels = [], []
    for label, name in [(0, "no"), (1, "yes")]:
        for p in sorted((data_dir / name).glob("*")):
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                paths.append(str(p))
                labels.append(label)
    if not paths:
        raise FileNotFoundError(f"No images found under {data_dir}/yes and {data_dir}/no")
    return np.array(paths), np.array(labels)


def make_ds(paths, labels, img_size, batch, training):
    def load(path, label):
        img = tf.io.read_file(path)
        img = tf.io.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.resize(img, (img_size, img_size))
        return tf.cast(img, tf.float32), tf.cast(label, tf.float32)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(len(paths), seed=SEED)
    ds = ds.map(load, num_parallel_calls=AUTOTUNE).batch(batch)
    return ds.prefetch(AUTOTUNE)


# --------------------------------------------------------------- models
# Both models take RAW pixels (0-255, shape HxWx3) so the API needs no
# separate preprocessing step.
def augmentation():
    return keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.08),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
    ], name="augmentation")


def build_custom_cnn(img_size):
    inp = keras.Input((img_size, img_size, 3))
    x = augmentation()(inp)
    x = layers.Rescaling(1.0 / 255)(x)
    for filters in (32, 64, 128, 256):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling2D()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inp, out, name="custom_cnn")


def build_mobilenet(img_size):
    base = keras.applications.MobileNetV2(
        input_shape=(img_size, img_size, 3), include_top=False, weights="imagenet")
    base.trainable = False
    inp = keras.Input((img_size, img_size, 3))
    x = augmentation()(inp)
    x = keras.applications.mobilenet_v2.preprocess_input(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inp, out, name="mobilenetv2"), base


def compile_model(model, lr):
    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc"),
                 keras.metrics.Recall(name="recall"),
                 keras.metrics.Precision(name="precision")])


def callbacks():
    return [
        keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=8,
                                      restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3),
    ]


# ------------------------------------------------------------ evaluate
def predict_probs(model, ds):
    y_true = np.concatenate([y.numpy() for _, y in ds])
    y_prob = model.predict(ds, verbose=0).ravel()
    return y_true, y_prob


def plot_history(histories, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name, h in histories.items():
        axes[0].plot(h["val_auc"], label=name)
        axes[1].plot(h["val_loss"], label=name)
    axes[0].set_title("Validation AUC"); axes[1].set_title("Validation loss")
    for a in axes:
        a.set_xlabel("epoch"); a.legend()
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--out_dir", default="models")
    args = ap.parse_args()

    tf.keras.utils.set_random_seed(SEED)
    out = Path(args.out_dir); out.mkdir(exist_ok=True)

    # Stratified 70 / 15 / 15 split on file paths (no leakage between sets)
    paths, labels = collect_files(Path(args.data_dir))
    print(f"Images: {len(paths)}  tumor={labels.sum()}  no_tumor={(labels == 0).sum()}")
    p_tr, p_tmp, y_tr, y_tmp = train_test_split(
        paths, labels, test_size=0.30, stratify=labels, random_state=SEED)
    p_va, p_te, y_va, y_te = train_test_split(
        p_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=SEED)

    train_ds = make_ds(p_tr, y_tr, args.img_size, args.batch, True)
    val_ds = make_ds(p_va, y_va, args.img_size, args.batch, False)
    test_ds = make_ds(p_te, y_te, args.img_size, args.batch, False)

    # Class weights to handle the tumor / no-tumor imbalance
    n, n_pos = len(y_tr), y_tr.sum()
    class_weight = {0: n / (2 * (n - n_pos)), 1: n / (2 * n_pos)}

    results, histories, trained = {}, {}, {}

    # ---- 1) custom CNN
    cnn = build_custom_cnn(args.img_size)
    compile_model(cnn, 1e-3)
    h = cnn.fit(train_ds, validation_data=val_ds, epochs=args.epochs,
                class_weight=class_weight, callbacks=callbacks(), verbose=2)
    histories["custom_cnn"], trained["custom_cnn"] = h.history, cnn

    # ---- 2) MobileNetV2: train head, then fine-tune top layers
    mnet, base = build_mobilenet(args.img_size)
    compile_model(mnet, 1e-3)
    h1 = mnet.fit(train_ds, validation_data=val_ds, epochs=args.epochs,
                  class_weight=class_weight, callbacks=callbacks(), verbose=2)
    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False
    compile_model(mnet, 1e-5)
    h2 = mnet.fit(train_ds, validation_data=val_ds, epochs=args.epochs,
                  class_weight=class_weight, callbacks=callbacks(), verbose=2)
    histories["mobilenetv2"] = {k: h1.history[k] + h2.history[k] for k in h1.history}
    trained["mobilenetv2"] = mnet

    plot_history(histories, out / "training_curves.png")

    # ---- pick best on VALIDATION AUC (test set stays untouched until the end)
    for name, m in trained.items():
        yt, yp = predict_probs(m, val_ds)
        results[name] = roc_auc_score(yt, yp)
        print(f"[val] {name}: AUC={results[name]:.4f}")
    best_name = max(results, key=results.get)
    best = trained[best_name]
    print(f"\nBest model: {best_name}")

    # ---- final report on held-out TEST set
    yt, yp = predict_probs(best, test_ds)
    yhat = (yp >= args.threshold).astype(int)
    report = classification_report(yt, yhat, target_names=["no_tumor", "tumor"],
                                   output_dict=True, zero_division=0)
    print(classification_report(yt, yhat, target_names=["no_tumor", "tumor"], zero_division=0))
    cm = confusion_matrix(yt, yhat)
    ConfusionMatrixDisplay(cm, display_labels=["no_tumor", "tumor"]).plot(cmap="Blues")
    plt.savefig(out / "confusion_matrix.png", dpi=120); plt.close()

    best.save(out / "best_model.keras")
    meta = {
        "model_name": best_name,
        "img_size": args.img_size,
        "threshold": args.threshold,
        "labels": {"0": "no_tumor", "1": "tumor"},
        "val_auc": results,
        "test_auc": float(roc_auc_score(yt, yp)),
        "test_report": report,
    }
    (out / "model_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Saved model + metadata to {out}/")


if __name__ == "__main__":
    main()
