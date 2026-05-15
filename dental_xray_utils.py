from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
DEFAULT_IMAGE_SIZE = (224, 224)


DISEASE_KNOWLEDGE = {
    "caries": {
        "summary": "Radiographic pattern is most consistent with dental caries.",
        "clinical_note": "Caries can appear as radiolucent defects in enamel or dentin and may need restorative treatment depending on extent.",
    },
    "deep caries": {
        "summary": "Radiographic pattern is most consistent with deep caries.",
        "clinical_note": "Deep caries suggests advanced decay close to the pulp and may require urgent restorative or endodontic assessment.",
    },
    "impacted tooth": {
        "summary": "Radiographic pattern is most consistent with an impacted tooth.",
        "clinical_note": "Impacted teeth may be unerupted or malpositioned and can contribute to pain, crowding, or adjacent tooth damage.",
    },
    "periapical lesion": {
        "summary": "Radiographic pattern is most consistent with a periapical lesion.",
        "clinical_note": "Periapical lesions may indicate infection or inflammation near the root apex and commonly require endodontic evaluation.",
    },
    "periodontal disease": {
        "summary": "Radiographic pattern is most consistent with periodontal disease.",
        "clinical_note": "Periodontal disease often appears with alveolar bone loss and requires periodontal examination and cleaning or advanced care.",
    },
    "root resorption": {
        "summary": "Radiographic pattern is most consistent with root resorption.",
        "clinical_note": "Root resorption reflects loss of root structure and may require specialist review to determine cause and treatment urgency.",
    },
    "missing teeth": {
        "summary": "Radiographic pattern is most consistent with missing teeth.",
        "clinical_note": "Missing teeth should be correlated with dental history and may require prosthetic, orthodontic, or surgical planning.",
    },
}


@dataclass
class DatasetBundle:
    train_ds: tf.data.Dataset
    val_ds: tf.data.Dataset
    class_names: list[str]
    train_count: int
    val_count: int
    data_root: Path


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_SUFFIXES


def normalize_label(label: str) -> str:
    cleaned = label.strip().lower().replace("_", " ").replace("-", " ")
    while "  " in cleaned:
        cleaned = cleaned.replace("  ", " ")
    return cleaned


def discover_classification_root(dataset_dir: str | Path) -> Path:
    dataset_path = Path(dataset_dir).expanduser().resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_path}")

    candidates: list[tuple[int, Path]] = []
    for directory in [dataset_path] + [p for p in dataset_path.rglob("*") if p.is_dir()]:
        class_dirs = [p for p in directory.iterdir() if p.is_dir()]
        if len(class_dirs) < 2:
            continue

        image_total = 0
        valid_class_dirs = 0
        for class_dir in class_dirs:
            count = sum(1 for file_path in class_dir.rglob("*") if file_path.is_file() and is_image_file(file_path))
            if count > 0:
                valid_class_dirs += 1
                image_total += count

        if valid_class_dirs >= 2 and image_total > 0:
            candidates.append((image_total, directory))

    if not candidates:
        raise RuntimeError(
            "Could not find a classification root with at least two class subfolders containing images."
        )

    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def count_images_per_class(root_dir: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for class_dir in sorted([p for p in root_dir.iterdir() if p.is_dir()]):
        counts[class_dir.name] = sum(1 for file_path in class_dir.rglob("*") if file_path.is_file() and is_image_file(file_path))
    return {name: count for name, count in counts.items() if count > 0}


def build_datasets(
    dataset_dir: str | Path,
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    batch_size: int = 16,
    validation_split: float = 0.2,
    seed: int = 42,
) -> DatasetBundle:
    data_root = discover_classification_root(dataset_dir)
    counts = count_images_per_class(data_root)
    total_images = sum(counts.values())
    if total_images < 10:
        raise RuntimeError(f"Not enough images found to train a model. Found {total_images}.")

    common_args = dict(
        directory=str(data_root),
        labels="inferred",
        label_mode="int",
        color_mode="rgb",
        image_size=image_size,
        batch_size=batch_size,
        validation_split=validation_split,
        seed=seed,
    )

    train_ds = tf.keras.utils.image_dataset_from_directory(
        subset="training",
        shuffle=True,
        **common_args,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        subset="validation",
        shuffle=False,
        **common_args,
    )

    class_names = list(train_ds.class_names)
    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(autotune)
    val_ds = val_ds.prefetch(autotune)

    val_count = max(1, math.floor(total_images * validation_split))
    train_count = total_images - val_count

    return DatasetBundle(
        train_ds=train_ds,
        val_ds=val_ds,
        class_names=class_names,
        train_count=train_count,
        val_count=val_count,
        data_root=data_root,
    )


def compute_class_weights(root_dir: Path, class_names: Iterable[str]) -> dict[int, float]:
    counts = count_images_per_class(root_dir)
    total = sum(counts.values())
    weights: dict[int, float] = {}
    normalized_counts = {normalize_label(name): count for name, count in counts.items()}
    normalized_names = {normalize_label(name): idx for idx, name in enumerate(class_names)}
    for label, idx in normalized_names.items():
        count = normalized_counts.get(label, 0)
        if count <= 0:
            continue
        weights[idx] = total / (len(normalized_names) * count)
    return weights


def build_model(num_classes: int, image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE) -> tf.keras.Model:
    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.03),
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.RandomContrast(0.1),
        ],
        name="augmentation",
    )

    try:
        base_model = tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights="imagenet",
            input_shape=(image_size[0], image_size[1], 3),
        )
    except Exception:
        # Fallback keeps training usable even when pretrained weights cannot be fetched.
        base_model = tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights=None,
            input_shape=(image_size[0], image_size[1], 3),
        )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(image_size[0], image_size[1], 3))
    x = augmentation(inputs)
    x = tf.keras.applications.efficientnet.preprocess_input(x)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="dental_xray_classifier")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def fine_tune_model(model: tf.keras.Model, learning_rate: float = 1e-4) -> tf.keras.Model:
    base_model = next((layer for layer in model.layers if isinstance(layer, tf.keras.Model)), None)
    if not isinstance(base_model, tf.keras.Model):
        return model

    base_model.trainable = True
    for layer in base_model.layers[:-20]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def load_and_prepare_image(image_path: str | Path, image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE) -> np.ndarray:
    image = tf.keras.utils.load_img(image_path, color_mode="rgb", target_size=image_size)
    image_array = tf.keras.utils.img_to_array(image)
    return np.expand_dims(image_array, axis=0)


def generate_report(
    image_path: str | Path,
    class_names: list[str],
    probabilities: np.ndarray,
    top_k: int = 3,
) -> dict:
    scores = probabilities.astype(float).tolist()
    ranked_indices = np.argsort(probabilities)[::-1][:top_k]
    primary_label = class_names[int(ranked_indices[0])]
    normalized_primary = normalize_label(primary_label)
    knowledge = DISEASE_KNOWLEDGE.get(
        normalized_primary,
        {
            "summary": f"Radiographic pattern is most consistent with {primary_label}.",
            "clinical_note": "Correlation with clinical findings is required before any treatment decision.",
        },
    )

    top_predictions = [
        {
            "label": class_names[int(idx)],
            "confidence": round(float(probabilities[int(idx)]), 4),
        }
        for idx in ranked_indices
    ]

    report_text = (
        f"AI analysis of panoramic X-ray `{Path(image_path).name}` suggests `{primary_label}` as the most likely finding "
        f"with confidence {top_predictions[0]['confidence']:.2%}. {knowledge['summary']} {knowledge['clinical_note']} "
        "This output is decision support only and must be reviewed by a qualified dental professional."
    )

    return {
        "image": str(Path(image_path).resolve()),
        "predicted_label": primary_label,
        "predicted_confidence": round(float(probabilities[int(ranked_indices[0])]), 4),
        "top_predictions": top_predictions,
        "all_scores": {class_names[i]: round(score, 4) for i, score in enumerate(scores)},
        "report": report_text,
    }


def evaluate_model(model: tf.keras.Model, val_ds: tf.data.Dataset, class_names: list[str]) -> dict:
    y_true: list[int] = []
    for _, labels in val_ds:
        y_true.extend(labels.numpy().tolist())

    predictions = model.predict(val_ds, verbose=0)
    y_pred = np.argmax(predictions, axis=1).tolist()

    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    matrix = confusion_matrix(y_true, y_pred).tolist()
    return {
        "classification_report": report,
        "confusion_matrix": matrix,
    }


def save_json(data: dict, output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, indent=2), encoding="utf-8")


def save_class_names(class_names: list[str], output_path: str | Path) -> None:
    save_json({"class_names": class_names}, output_path)


def load_class_names(labels_path: str | Path) -> list[str]:
    payload = json.loads(Path(labels_path).read_text(encoding="utf-8"))
    class_names = payload.get("class_names")
    if not isinstance(class_names, list) or not class_names:
        raise ValueError(f"Invalid labels file: {labels_path}")
    return [str(item) for item in class_names]


def predict_image_report(
    model: tf.keras.Model,
    class_names: list[str],
    image_path: str | Path,
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
) -> dict:
    image_batch = load_and_prepare_image(image_path=image_path, image_size=image_size)
    probabilities = model.predict(image_batch, verbose=0)[0]
    return generate_report(image_path=image_path, class_names=class_names, probabilities=probabilities)
